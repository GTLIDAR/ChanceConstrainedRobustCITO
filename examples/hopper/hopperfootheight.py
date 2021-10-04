"""


"""
import numpy as np
import decorators as deco
import matplotlib.pyplot as plt
import utilities as utils
import os, csv
from pydrake.all import PiecewisePolynomial
from systems.hopper.hopper import Hopper
from robusthoppingopt import RobustOptimizationOptions

def load_data_recursive(directory, filename = 'trajoptresults.pkl'):
    data = [utils.load(os.path.join(file, filename)) for file in utils.find_filepath_recursive(directory, filename)]
    return data

def get_hopper_foot_trajectory(hopper, x):
    # Calculate the collision point trajectories
    context = hopper.multibody.CreateDefaultContext()
    foot = np.zeros((3, x.shape[1]))
    for n in range(x.shape[1]):
        hopper.multibody.SetPositionsAndVelocities(context, x[:,n])    
        cpt = hopper.get_contact_points(context)
        foot[:, n] = (cpt[0][:,-1] + cpt[1][:,-1])/2.
    return foot
    
@deco.showable_fig
@deco.saveable_fig
def plot_foot_base_height(hopper, xtraj, label='data', fig=None, axs=None):
    t, x = utils.GetKnotsFromTrajectory(xtraj)
    foot = get_hopper_foot_trajectory(hopper, x)
    if fig is None and axs is None:
        fig, axs = plt.subplots(2,1)
    axs[0].plot(t, x[1,:], linewidth=1.5, label=label)
    axs[0].set_ylabel('Base Height (m)')
    axs[1].plot(t, foot[-1, :], linewidth=1.5, label=label)
    axs[1].set_ylabel('Foot Height (m)')
    axs[1].set_xlabel('Time (s)')
    return fig, axs

@deco.showable_fig
@deco.saveable_fig
def compare_foot_base_height(hopper, xlist, labels):
    fig, axs = plot_foot_base_height(hopper, xlist[0], label=labels[0], show=False)
    for x, label in zip(xlist[1:], labels[1:]):
        fig, axs = plot_foot_base_height(hopper, x, label, fig, axs, show=False)
    #Turn on the legend
    axs[0].legend()
    return fig, axs

@deco.showable_fig
@deco.saveable_fig
def plot_foot_height_bar_chart(foot_height, labels):
    # Setup the bar chart
    x = np.arange(len(labels))
    fig, ax = plt.subplots(1,1)
    rect1 = ax.bar(x, foot_height)
    # Add text labels
    ax.set_ylabel('Foot height (m)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    fig.tight_layout()

    return fig, ax

def compare_multiple_hoppers(files, savename):
    xlist = []
    labels = []
    for file in files:
        data = utils.load(file)
        xlist.append(PiecewisePolynomial.FirstOrderHold(data['time'], data['state']))
        labels.append(f"$\sigma$ = {data['sigma']}")
    hopper = Hopper()
    hopper.Finalize()
    compare_foot_base_height(hopper, xlist, labels, show=False, savename=savename)

def compare_ERM_footheights(directory, reffile=None, savename='FootHeightComparison.pdf'):
    if reffile is None:
        reffile = os.path.join("examples","hopper","reference_linear","strict_nocost","trajoptresults.pkl")
    file = 'trajoptresults.pkl'
    xlist = []
    labels = []
    # ERM data
    for pathname in utils.find_filepath_recursive(directory, file):
        fullpath = os.path.join(pathname, file)
        data = utils.load(fullpath)
        xlist.append(PiecewisePolynomial.FirstOrderHold(data['time'], data['state']))
        labels.append(f"$\sigma$ = {data['sigma']}")
    # Reference case
    data = utils.load(reffile)
    xlist.append(PiecewisePolynomial.FirstOrderHold(data['time'], data['state']))
    labels.append(f"Reference")
    savename = os.path.join(directory, savename)
    # Create the hopper
    hopper = Hopper()
    hopper.Finalize()
    compare_foot_base_height(hopper, xlist, labels, show=False, savename=savename)
    print(f"Comparison saved to {savename}")

def compare_chance_footheights(erm_dir, cc_dir, ref_dir = None):
    # Create the hopper
    hopper = Hopper()
    hopper.Finalize()
    
    # Initialize the lists
    erm_list = []
    erm_sigma = []
    cc_list = []
    cc_sigma = []
    cc_labels = []
    # Load the ERM Data
    filename = 'trajoptresults.pkl'
    for pathname in utils.find_filepath_recursive(erm_dir, filename):
        fullpath = os.path.join(pathname, filename)
        data = utils.load(fullpath)
        erm_list.append(PiecewisePolynomial.FirstOrderHold(data['time'], data['state']))
        erm_sigma.append(data['sigma'])
    # Load the CC Data
    for pathname in utils.find_filepath_recursive(cc_dir, filename):
        fullpath = os.path.join(pathname, filename)
        data = utils.load(fullpath)
        cc_list.append(PiecewisePolynomial.FirstOrderHold(data['time'], data['state']))
        cc_sigma.append(data['sigma'])
        cc_labels.append(f"$\\theta$ = {data['theta']}, $\\beta$={data['beta']}")
    # Load the reference case, if possible
    if ref_dir is not None:
        data = utils.load(os.path.join(ref_dir, filename))
        reference = PiecewisePolynomial.FirstOrderHold(data['time'], data['state'])
    # Filter out by the SIGMA value, and plot
    for n in range(len(erm_sigma)):
        sigma=erm_sigma[n]
        savestr = f'FootHeightComparison_Chance_Sigma{sigma:.0e}.png'
        sig_idx = [i for i, x in enumerate(cc_sigma) if x == sigma]
        cc_shortlist = [cc_list[i] for i in sig_idx]
        cc_shortlabel = [cc_labels[i] for i in sig_idx]
        cc_shortlabel.insert(0,"ERM")
        cc_shortlist.insert(0, erm_list[n])
        if ref_dir is not None:
            cc_shortlabel.insert(0,"Referece")
            cc_shortlist.insert(0, reference)
        compare_foot_base_height(hopper, cc_shortlist, cc_shortlabel, show=False, savename=os.path.join(cc_dir, savestr))
        print(f"Comparison saved to {os.path.join(cc_dir, savestr)}")

def common_sigma_data(ermdata, ccdata, ermsigmas, ccsigmas):
    # Find the sigmas in ccSig that are also in ermSig
    both = set(ermsigmas).intersection(ccsigmas)
    cc_idx = sorted([ccsigmas.index(x) for x in both])
    erm_idx = sorted([ermsigmas.index(x) for x in both])
    common_sigma = [ermsigmas[i] for i in erm_idx]
    # Filter down and combine with the ERM data
    ermdata = np.reshape(ermdata, (1, ermdata.shape[0]))
    common_data = np.concatenate([ermdata[:, erm_idx], ccdata[:, cc_idx]], axis=0)
    return common_data, common_sigma

def get_all_data(ermdir, ccdir, refdir, footfunc):
    hopper = Hopper()
    hopper.Finalize()
    filename = 'trajoptresults.pkl'
    # Reference data
    data = utils.load(os.path.join(refdir, filename))
    foot = get_hopper_foot_trajectory(hopper, data['state'])[-1,:]
    refdata = footfunc(foot)
    # ERM Data
    ermdataset = [data for data in load_data_recursive(ermdir)]
    ermsigmas = list(set([data['sigma'] for data in ermdataset]))
    ermsigmas = sorted(ermsigmas)
    ermdata = np.full((len(ermsigmas),), np.nan)
    for data in ermdataset:
        sindex = ermsigmas.index(data['sigma'])
        if data['success']:
            foot = get_hopper_foot_trajectory(hopper, data['state'])[-1, :]
            ermdata[sindex] = footfunc(foot)
    # Chance Data
    chancedataset = [data for data in load_data_recursive(ccdir)]
    ccsigma = list(set([data['sigma'] for data in chancedataset]))
    cctheta = list(set([data['theta'] for data in chancedataset]))
    ccsigma = sorted(ccsigma)
    cctheta = sorted(cctheta)
    ccdata = np.full((len(cctheta), len(ccsigma)), np.nan)
    for data in chancedataset:
        sindex = ccsigma.index(data['sigma'])
        tindex = cctheta.index(data['theta'])
        if data['success']:
            foot = get_hopper_foot_trajectory(hopper, data['state'])[-1,:]
            ccdata[tindex, sindex] = footfunc(foot)
    cclabels = [f'$\\theta={theta:0.1f}' for theta in cctheta]
    return refdata, ermdata, ccdata, ermsigmas, ccsigma, cclabels


def calculate_mean_footheight(erm_dir, cc_dir, ref_dir = None):
    refdata, ermdata, ccdata, ermsigmas, ccsigma, labels = get_all_data(erm_dir, cc_dir, ref_dir, np.average)
    ermccdata, common_sigma = common_sigma_data(ermdata, ccdata, ermsigmas, ccsigma)
    # Append the data together
    labels.insert(0, 'ERM')
    labels.insert(0, 'Reference')
    refdata = refdata * np.ones((1, ermccdata.shape[1]))
    alldata = np.concatenate([refdata, ermccdata], axis=0)
    # Make and save the figures
    for n in range(alldata.shape[1]):
        savename = os.path.join(cc_dir, f'MeanFootHeightComparison_sigma{common_sigma[n]:.0e}.pdf')
        plot_foot_height_bar_chart(alldata[:,n], labels, show=False, savename=savename)
        print(f"Figure saved to {savename}")

    # Write the data to a table
    foot_file = os.path.join(cc_dir, 'MeanFootHeight.csv')
    common_sigma.insert(0, 'Sigma')
    with open(foot_file, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(common_sigma)
        # Write each line thereafter
        for n in range(alldata.shape[0]):
            data = alldata[n].tolist()
            data.insert(0, labels[n])
            writer.writerow(data)
    print(f"Saved Mean foot height data to {foot_file}")

def calculate_peak_footheight(erm_dir, cc_dir, ref_dir = None):
    refdata, ermdata, ccdata, ermsigmas, ccsigma, labels = get_all_data(erm_dir, cc_dir, ref_dir, np.amax)
    ermccdata, common_sigma = common_sigma_data(ermdata, ccdata, ermsigmas, ccsigma)
    # Append the data together
    labels.insert(0, 'ERM')
    labels.insert(0, 'Reference')
    refdata = refdata * np.ones((1, ermccdata.shape[1]))
    alldata = np.concatenate([refdata, ermccdata], axis=0)
    # Make and save the figures
    for n in range(alldata.shape[1]):
        savename = os.path.join(cc_dir, f'MaxFootHeightComparison_sigma{common_sigma[n]:.0e}.pdf')
        plot_foot_height_bar_chart(alldata[:,n], labels, show=False, savename=savename)
        print(f"Figure saved to {savename}")

    # Write the data to a table
    foot_file = os.path.join(cc_dir, 'MaxFootHeight.csv')
    common_sigma.insert(0, 'Sigma')
    with open(foot_file, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(common_sigma)
        # Write each line thereafter
        for n in range(alldata.shape[0]):
            data = alldata[n].tolist()
            data.insert(0, labels[n])
            writer.writerow(data)
    print(f"Saved Mean foot height data to {foot_file}")


def main():
    directory = os.path.join('examples','hopper','robust_erm_linear_hotfix_tol1e-8 _scale1_erm')
    file = 'trajoptresults.pkl'
    for pathname in utils.find_filepath_recursive(directory, file):
        print(f"Working on file {os.path.join(pathname, file)}")
        data = utils.load(os.path.join(pathname, file))
        xtraj = PiecewisePolynomial.FirstOrderHold(data['time'], data['state'])
        hopper = Hopper()
        hopper.Finalize()
        plot_foot_base_height(hopper, xtraj, show=False, savename=os.path.join(pathname, 'FootHeight.png'))






if __name__ == "__main__":
    erm_dir = os.path.join('examples','hopper','robust_nonlinear','erm_1e5','success')
    cc_dir = os.path.join('examples','hopper','robust_nonlinear','cc_erm_1e5_mod','success')
    ref_dir = os.path.join('examples','hopper','reference_linear','strict_nocost')
    compare_chance_footheights(erm_dir, cc_dir, ref_dir)
    # compare_ERM_footheights(os.path.join(directory = os.path.join('examples','hopper','robust_nonlinear','erm_1e5','success')))