#!/usr/bin/env python

import datetime
import argparse

from cubesat.cubesat import CubeSat

__author__ = 'Teruaki Enoto'
__version__ = '0.02'
# v0.02 : 2020-08-02 : COR map added.
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
	parser.add_argument('--start', '-s', type=str, default=None,
		help='input start UTC datetime of the orbit (e.g., 2023-12-10_12:00:00)')
	parser.add_argument('--end', '-e', type=str, default=None, 
		help='input end UTC date d of the orbit.')		
	parser.add_argument('--setup_yamlfile', '-y', type=str, default="data/ninjasat_setup.yaml", 
		help='setup yamlfile for the satellite')		
	parser.add_argument('--timebin_minute', '-t', type=float, default=1.0,
		help='time bin for plotting.')	
	parser.add_argument('--foutname_base', '-o', type=str, default="orbit",
		help='output filename base for csv and pdf')					
	return parser

def main(args=None):
	parser = get_parser()
	args = parser.parse_args(args)

	ninjasat = CubeSat(setup_yamlfile=args.setup_yamlfile)
	ninjasat.plot_cutoff_rigidity_map()		
	ninjasat.plot_orbit(
		start_date_utc=args.start,
		end_date_utc=args.end,
		timebin_minute=args.timebin_minute,
		foutname_base=args.foutname_base)

if __name__=="__main__":
	main()
