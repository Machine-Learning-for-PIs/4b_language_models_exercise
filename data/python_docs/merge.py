""" Merge the python docs into a single file."""

import os
import pathlib


if __name__ == '__main__':

	with open("input.txt", "w") as write_to_input_file:
		for root, folder, docfiles in os.walk("./python-3.13-docs-text"):
			for docfile in docfiles:
				if docfile.split(".")[-1] == 'txt':
					path = pathlib.Path(f"{root}/{docfile}")
					with open(path, "r") as read_file:
						lines = read_file.readlines()
						write_to_input_file.writelines(lines)
						write_to_input_file.write(" ")


