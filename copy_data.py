import os
import shutil
import sys
from tqdm import tqdm
def count_files(src, exclude_patterns):
    total_files = 0
    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        if rel_root == '.':
            rel_root = ''

        # Exclude directories whose paths include any of the exclude patterns
        dirs[:] = [d for d in dirs if not any(pattern in os.path.join(rel_root, d) for pattern in exclude_patterns)]

        for file in files:
            rel_file = os.path.join(rel_root, file)
            if any(pattern in rel_file for pattern in exclude_patterns):
                continue
            total_files += 1
    return total_files

def collect_files_and_dirs(src, exclude_patterns):
    """Collect all files and directories to be copied, excluding based on patterns."""
    file_list = []
    for root, dirs, files in os.walk(src):
        # Exclude directories based on patterns
        dirs[:] = [d for d in dirs if not any(pattern in os.path.join(root, d) for pattern in exclude_patterns)]

        for file in files:
            rel_root = os.path.relpath(root, src)
            if rel_root == '.':
                rel_root = ''
            rel_file = os.path.join(rel_root, file)
            if not any(pattern in rel_file for pattern in exclude_patterns):
                file_list.append((root, file))  # Store root and file info
    return file_list

def copy_with_exclude(src, dst, exclude_patterns):
    """
    Copy files from src to dst, excluding any directories or files whose paths include any of the exclude_patterns.
    Uses tqdm to show progress.
    """
    # Collect all the files and directories to copy
    file_list = collect_files_and_dirs(src, exclude_patterns)
    total_files = len(file_list)
    print(f"Total files to copy: {total_files}")

    # Initialize tqdm progress bar
    with tqdm(total=total_files, unit="file", desc="Copying files") as progress_bar:
        for root, file in file_list:
            rel_root = os.path.relpath(root, src)
            if rel_root == '.':
                rel_root = ''

            dest_root = os.path.join(dst, rel_root)

            if not os.path.exists(dest_root):
                os.makedirs(dest_root)

            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)
            # Check if the file exists in the destination directory
            if os.path.exists(dest_file):
                progress_bar.update(1)
                continue
            try:
                shutil.copy2(src_file, dest_file)
                progress_bar.set_description(f"Copying {src_file}")  # Show current file in tqdm
                progress_bar.update(1)  # Update the progress bar
            except Exception as e:
                print(f"Error copying {src_file} to {dest_file}: {e}")

    print(f"Finished copying {total_files} files.")
# def copy_with_exclude(src, dst, exclude_patterns):
#     """
#     Copy files from src to dst, excluding any directories or files whose paths include any of the exclude_patterns.
#     Tracks the progress of the copying process.
#
#     :param src: Source directory path.
#     :param dst: Destination directory path.
#     :param exclude_patterns: List of substrings to search for in paths to exclude.
#     """
#     # Count total files to copy
#     print("Counting files to copy...")
#     total_files = count_files(src, exclude_patterns)
#
#     print(f"Total files to copy: {total_files}")
#
#     for root, dirs, files in os.walk(src):
#         # Compute the relative path from the source directory
#         rel_root = os.path.relpath(root, src)
#         if rel_root == '.':
#             rel_root = ''
#
#         # Exclude directories whose paths include any of the exclude patterns
#         dirs[:] = [d for d in dirs if not any(pattern in os.path.join(rel_root, d) for pattern in exclude_patterns)]
#
#         # Now process files in the current directory
#         dest_root = os.path.join(dst, rel_root)
#         if not os.path.exists(dest_root):
#             try:
#                 os.makedirs(dest_root)
#             except Exception as e:
#                 print(f"Error creating directory {dest_root}: {e}")
#                 continue
#
#         for file in files:
#             rel_file = os.path.join(rel_root, file)
#             # Check if the file should be excluded
#             if any(pattern in rel_file for pattern in exclude_patterns):
#                 continue
#             src_file = os.path.join(root, file)
#             dest_file = os.path.join(dest_root, file)
#             try:
#                 shutil.copy2(src_file, dest_file)
#             except Exception as e:
#                 print(f"Error copying {src_file} to {dest_file}: {e}")
#
#     print("Finished copying")

if __name__ == '__main__':
    # Usage: python script.py <source_dir> <destination_dir>
    # if len(sys.argv) < 3:
    #     print("Usage: python script.py <source_dir> <destination_dir>")
    #     sys.exit(1)

    #src_dir = sys.argv[1]
    #dst_dir = sys.argv[2]
    src_dir = "/media/dofri/OBSERVATIONS/VST_BUFFER/2022-01"
    # src_dir = "/home/dofri/epfl/semester_project/data"
    dst_dir = "/media/dofri/MedTina/VST_BUFFER/2022-01"
    exclude_patterns = ['L0_RAW', 'L1_DETECTION']  # Patterns to exclude

    copy_with_exclude(src_dir, dst_dir, exclude_patterns)