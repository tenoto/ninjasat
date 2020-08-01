#!/usr/bin/env python

import os
import argparse

__author__ = 'Teruaki Enoto'
__version__ = '0.01'
# v0.01 : 2020-08-01 : original version

def get_parser():
	"""
	Creates a new argument parser.
	"""
	parser = argparse.ArgumentParser('plot_ninjasat_orbit.py',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		description="""
Plot NinjaSat ortbit at the specified date.
		"""
		)
	version = '%(prog)s ' + __version__
	parser.add_argument('--version', '-v', action='version', version=version,
		help='show version of this command')
	parser.add_argument('--date', '-d', type=str, required=True, 
		help='input start date of the orbit')	
	return parser

def plot_ninjasat_orbit(obsid, yyyy_mm):
	print("plot_ninjasat_orbit")

def main(args=None):
	parser = get_parser()
	args = parser.parse_args(args)

	cmd = run_wget(args.obsid, args.yyyy_mm)

if __name__=="__main__":
	main()