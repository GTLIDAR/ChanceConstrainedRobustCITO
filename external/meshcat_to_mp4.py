"""
Converts .tar files downloaded from meshcat to .mp4 files

This file does NOT depend on Drake, and can (and should) be used outside the pyCITO container

Luke Drnach
July 27, 2021
"""

import os, sys, tarfile, getopt

def recursive_tar_generator(source):
    for path, dir, files in os.walk(source):
        for file in files:
            if os.path.splitext(file)[1] == ".tar":
                yield os.path.join(path, file)

def extract_folder(directory):
    """Extract a compressed directory to another directory of the same name"""
    target = os.path.splitext(directory)[0]
    with tarfile.open(directory, 'r') as tar:
        tar.extractall(target)
    return target

def meshcat_to_mp4(directory, savename=None):
    """ Convert meshcat directory to mp4 """
    # Change directories
    pwd = os.getcwd()
    os.chdir(directory)
    # Run ffmpeg
    if savename is None:
        savename = os.path.basename(directory)
    command = f"ffmpeg -r 60 -i %07d.png -vcodec libx264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -preset slow -crf 18 {savename}.mp4"
    print(command)
    os.system(command)
    # Create a GIF as well
    gifcommand = f"ffmpeg -i %07d.png -vf fps=20,scale=720:-1 {savename}.gif"
    print(gifcommand)
    os.system(gifcommand)
    # Revert directories
    os.chdir(pwd)
    return savename

def move_movie(source, target, savename=None):
    """Move the movie to another directory"""
    if savename is None:
        savename = os.path.basename(source)
    savename = os.path.splitext(savename)[0]
    source_mp4 = os.path.join(source, savename) + ".mp4"
    if os.path.isdir(target) is False:
        os.makedirs(target)
    target_mp4 = os.path.join(target, savename) 
    if os.path.isfile(target_mp4):
        os.remove(target_mp4)
    target_mp4 += ".mp4"
    # Move the file
    os.rename(source_mp4, target_mp4)
    # If there is a gif, move that as well
    source_gif = os.path.join(source, savename) + ".gif"
    if os.path.isfile(source_gif):
        target_gif = os.path.join(target, savename) + ".gif"
        os.rename(source_gif, target_gif)
    return target
        
def convert_meshcat_and_move(source, target):
    """Convert a meshcat folder to video and move the video to the target directory"""
    directory = extract_folder(source)
    filename = meshcat_to_mp4(directory) + ".mp4"
    destination = move_movie(directory, target, savename=filename)
    print(f"Meshcat movie saved in {destination}")

def batch_convert_files(directory, target):
    for file in recursive_tar_generator(directory):
        subdir = os.path.splitext(file.replace(directory, ''))[0]
        if subdir[0] == os.path.sep:
            subdir = subdir[1:]
        target_dir = os.path.join(target, subdir)
        print(f"Converting {dir}")
        convert_meshcat_and_move(file, target_dir)

def main(args):
    infile = None
    outfile = None
    try:
        opts, args = getopt.getopt(args, "hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print("meshcat_to_mp4.py -i <input directory> -o <target directory>")
    for opt, args in opts:
        if opt == "-h":
            print("meshcat_to_mp4.py -i <input directory> -o <target directory>")
            sys.exit()
        elif opt in ("-i", "--infile"):
            infile = opt
        elif opt in ("-o", "--outfile"):
            outfile = opt
    if infile is None or outfile is None:
        print("input and output filenames required")
        sys.exit()
    batch_convert_files(infile, outfile)

if __name__ == "__main__":
    infile = "/home/ldrnach3/Downloads/hopper"
    outfile = "/home/ldrnach3/Projects/pyCITO/examples/hopper/reference_linear"
    batch_convert_files(infile, outfile)
    #main(sys.argv[1:])