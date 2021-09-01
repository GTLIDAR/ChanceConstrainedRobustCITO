import os

def main(source, target):
    for path, dir, files in os.walk(source):
        for file in files:
            if file == "trajoptresults.pkl":
                new_dir = path.replace(source, '')
                new_dir = os.path.join(target, new_dir[1:])
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir)
                    print(f"Making directory: {new_dir}")
    print('finished')

if __name__ == "__main__":
    source = '/home/ldrnach3/Projects/pyCITO/examples/a1/runs/Jul-29-2021'
    target = '/home/ldrnach3/Downloads/Jul-29-2021'
    main(source, target)