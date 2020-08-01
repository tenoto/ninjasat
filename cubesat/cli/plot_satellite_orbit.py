#!/usr/bin/env python

import os
import datetime
import argparse
import astropy 
import pandas
import numpy as np

import cartopy.crs as ccrs

import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

from cubesat.cubesat import CubeSat

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
	parser.add_argument('--start', '-s', type=str, default=None,
		help='input start UTC datetime of the orbit (e.g., 2023-12-10_12:00:00)')
	parser.add_argument('--end', '-e', type=str, default=None, 
		help='input end UTC date d of the orbit.')		
	parser.add_argument('--setupy_yamlfile', '-y', type=str, default="data/ninjasat_setup.yaml", 
		help='setup yamlfile for the satellite')			
	return parser

def plot_orbit(setupy_yamlfile,start_date_utc,end_date_utc,timebin_minute=1.):
	print("---plot_ninjasat_orbit")

	ninjasat = CubeSat(setupy_yamlfile)

	if start_date_utc is None:
		#start_date_utc = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')		
		start_date_utc = datetime.datetime.now()
	else:
		start_date_utc = datetime.datetime.strptime(start_date_utc, '%Y-%m-%d_%H:%M:%S')
	print("Start date (UTC): {}".format(start_date_utc))

	if end_date_utc is None:
		end_date_utc = start_date_utc + datetime.timedelta(minutes=100)
	else:
		end_date_utc = datetime.datetime.strptime(end_date_utc, '%Y-%m-%d_%H:%M:%S')		
	print("Start date (UTC): {}".format(end_date_utc))

	list_time = []
	list_lon = []
	list_lat = []
	list_elevation = []	
	elapsed_time_minute = 0
	utc_time = start_date_utc
	while utc_time < end_date_utc:
		utc_time = start_date_utc + datetime.timedelta(minutes=elapsed_time_minute)
		lon, lat, alt = ninjasat.orbit.get_lonlatalt(utc_time)
		azimuth, elevation = ninjasat.orbit.get_observer_look(utc_time,25.2797,54.6872,112)
		list_time.append(utc_time)
		list_lon.append(lon)
		list_lat.append(lat)
		list_elevation.append(elevation)

		elapsed_time_minute += timebin_minute

	print(np.array(list_elevation)>10.0)

	fig = plt.figure(figsize=(12,8))
	ax = plt.axes(projection=ccrs.PlateCarree())	
	ax.stock_img()
	ax.coastlines(resolution='110m')

	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
		linewidth=2, color='gray', alpha=0.5, linestyle='--')
	#gl.xlabels_top = False
	#gl.ylabels_left = False
	#gl.xlines = False
	#gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
	#gl.xformatter = LONGITUDE_FORMATTER
	#gl.yformatter = LATITUDE_FORMATTER
	#gl.xlabel_style = {'size': 15, 'color': 'gray'}
	#gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

	ax.plot(list_lon,list_lat,".",color='#FF5733')	
	ax.plot(25.2797,54.6872,"*",markersize=20,color='r')	
	#	"Vilnius", latitude_deg=54.6872, longitude_deg=25.2797, elevation_m=112)
	plt.title("TLE:%s Date:%s to %s" % (
		ninjasat.param["tle_name"],
		start_date_utc.strftime('%Y-%m-%d-%H:%M:%S'),
		end_date_utc.strftime('%Y-%m-%d-%H:%M:%S')))
	ax.set_xlim(-180.0,180.0)
	ax.set_ylim(-90.0,90.0)	


	plt.savefig("orbit.pdf",bbox_inches='tight')

	"""
	fig = plt.figure(figsize=(12,8))
	ax = fig.add_subplot(111,aspect='equal')
	ax.plot(list_lon,list_lat,".")
	ax.set_xlim(-180.0,180.0)
	ax.set_ylim(-90.0,90.0)
	plt.savefig("test.pdf",bbox_inches='tight')
	"""

	#m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
	#	llcrnrlon=0,urcrnrlon=360,resolution='c')
	#x, y = m(lon2, lat2)

	#print(ninjasat.predictor.get_position(datetime.datetime.now()))
	#print(ninjasat.get_ecef_location(datetime.datetime.now()))
	#print(ninjasat.get_lla_location(datetime.datetime.now()))


def main(args=None):
	parser = get_parser()
	args = parser.parse_args(args)
	plot_orbit(setupy_yamlfile=args.setupy_yamlfile,start_date_utc=args.start,end_date_utc=args.end)

if __name__=="__main__":
	main()