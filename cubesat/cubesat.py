# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yaml 
import datetime
import ephem

from pyorbital.orbital import Orbital
from pyorbital.tlefile import Tle

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

COLUMN_KEYWORDS = ['utc_time','longitude_deg','latitude_deg','altitude_km','sun_elevation_deg','sat_elevation_deg','sat_azimuth_deg','cutoff_rigidity_GV']

class CubeSat():
	def __init__(self,setup_yamlfile):
		self.param = yaml.load(open(setup_yamlfile),Loader=yaml.SafeLoader)

		self.orbital_tle = Tle(self.param['tle_name'],self.param['tle_file'])
		self.orbital_orbit = Orbital(self.param['tle_name'], line1=self.orbital_tle.line1, line2=self.orbital_tle.line2)

		self.ephem_sun = ephem.Sun()
		self.ephem_sat = ephem.Observer()

		self.longitude_deg = None # 経度
		self.latitude_deg = None # 緯度
		self.altitude_km = None
		self.sun_elevation_deg = None
		self.sat_elevation_deg = None
		self.sat_azimuth_deg = None		

		self.track_dict = {}
		for keyword in COLUMN_KEYWORDS:
			self.track_dict[keyword] = []

		self.set_cutoff_rigidity_map()
		#self.set_hvoff_region()

	def get_position(self,utc_time):
		"""
		return satellite position: longitude (deg), latitude (deg), and altitude (km)
		"""
		return self.orbital_orbit.get_lonlatalt(utc_time)

	def set_position(self,utc_time):
		self.longitude_deg, self.latitude_deg, self.altitude_km = self.get_position(utc_time)

		# Sun elevation 
		# https://stackoverflow.com/questions/43299500/pandas-convert-datetime-timestamp-to-whether-its-day-or-night
		self.ephem_sat.date = utc_time
		self.ephem_sat.lat, self.ephem_sat.lon, self.ephem_sat.elevation = np.radians(self.latitude_deg), np.radians(self.longitude_deg), self.altitude_km*1000.0
		self.ephem_sun.compute(self.ephem_sat)
		self.sun_elevation_deg = np.degrees(self.ephem_sun.alt)

		# Satellite elevation from the station
		self.sat_azimuth_deg, self.sat_elevation_deg = self.orbital_orbit.get_observer_look(
			utc_time,
			self.param['ground_station_longitude_deg'],
			self.param['ground_station_latitude_deg'],
			self.param['ground_station_altitude_m'])

		# get cutoff rigidity
		self.cutoff_rigidity_GV = self.get_cutoff_rigidity(longitude_deg=self.longitude_deg,latitude_deg=self.latitude_deg)

		# add to track
		self.append_ro_track(utc_time)

	def append_ro_track(self,utc_time):
		for keyword in COLUMN_KEYWORDS:
			if keyword == 'utc_time':
				self.track_dict['utc_time'].append(utc_time)
			else:
				self.track_dict[keyword].append(getattr(self, keyword))

	def write_to_csv(self,foutname_csv):
		self.df = pd.DataFrame(self.track_dict,columns=COLUMN_KEYWORDS)
		print(self.df)
		self.df.to_csv("%s" % foutname_csv) 

	def plot_orbit(self,start_date_utc,end_date_utc,
		timebin_minute=1.,foutname_base='orbit'):

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

		elapsed_time_minute = 0
		utc_time = start_date_utc
		while utc_time < end_date_utc:
			utc_time = start_date_utc + datetime.timedelta(minutes=elapsed_time_minute)
			self.set_position(utc_time)
			elapsed_time_minute += timebin_minute

		self.write_to_csv(foutname_base+'.csv')

		fig = plt.figure(figsize=(12,8))
		ax = plt.axes(projection=ccrs.PlateCarree())	
		ax.stock_img()
		ax.coastlines(resolution='110m')

		plt.title("TLE:%s Date:%s to %s" % (
			self.param["tle_name"],
			start_date_utc.strftime('%Y-%m-%d-%H:%M:%S'),
			end_date_utc.strftime('%Y-%m-%d-%H:%M:%S')))

		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		#gl.xformatter = LONGITUDE_FORMATTER
		#gl.yformatter = LATITUDE_FORMATTER
		#gl.xlabel_style = {'size': 15, 'color': 'gray'}
		#gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)		
		ax.set_xlabel('Longitude (deg)')
		ax.set_ylabel('Latitude (deg)')		
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])	

		ax.plot(
			self.df["longitude_deg"][self.df["sun_elevation_deg"]>=0.0],
			self.df["latitude_deg"][self.df["sun_elevation_deg"]>=0.0],
			".",color='#FF5733')	
		ax.plot(
			self.df["longitude_deg"][self.df["sun_elevation_deg"]<0.0],
			self.df["latitude_deg"][self.df["sun_elevation_deg"]<0.0],
			".",color='#272889')	
		ax.scatter(
			self.df["longitude_deg"][self.df["sat_elevation_deg"]>=20.0],
			self.df["latitude_deg"][self.df["sat_elevation_deg"]>=20.0],
			facecolor='None', edgecolors='#278938')
		ax.plot(
			self.param['ground_station_longitude_deg'],
			self.param['ground_station_latitude_deg'],
			"*",markersize=20,color='r')	

		cormap_contour = ax.contour(
			self.cormap_longitude_centers, 
			self.cormap_latitude_centers, 
			self.cutoff_rigidity_map_T,
			levels=6, colors=['gray'], linewidths=1.2, 
			transform=ccrs.PlateCarree())
		cormap_contour.clabel(fmt='%1.1f', fontsize=12)

		plt.savefig("%s.pdf" % foutname_base,bbox_inches='tight')

	def set_cutoff_rigidity_map(self):
		print("--set_cutoff_rigidity_map")
		print('file_cutoffrigidity: {}'.format(self.param['file_cutoffrigidity']))

		self.df_cor = pd.read_csv(self.param['file_cutoffrigidity'],
			skiprows=1,delim_whitespace=True,
			names=['latitude_deg','longitude_deg','cutoff_rigidity'],
			dtype={'latitude_deg':'float64','longitude_deg':'float64','cutoff_rigidity':'float64'})
		self.cutoff_rigidity_map, self.cormap_longitude_edges, self.cormap_latitude_edges = np.histogram2d([],[],
			bins=[self.param["cormap_lon_nbin"],self.param["cormap_lat_nbin"]],
			range=[[-180.,180.],[-90.,90.]])	
		self.df_cor["longitude_index"] = np.digitize(self.df_cor['longitude_deg'], self.cormap_longitude_edges)
		self.df_cor["longitude_index"] = pd.to_numeric(self.df_cor["longitude_index"],downcast='signed')
		self.df_cor["latitude_index"] = np.digitize(self.df_cor['latitude_deg'], self.cormap_latitude_edges)
		self.df_cor["latitude_index"] = pd.to_numeric(self.df_cor["latitude_index"],downcast='signed')		
		"""
		H: two dimentional matrix 
		x: longitude
		y: latitude 
		
		H[0][2]		...
		H[0][1]		H[1][1]		...
		H[0][0]		H[1][0]		...		
		"""

		for index, row in self.df_cor.iterrows():
			i = int(row['longitude_index'])-1
			j = int(row['latitude_index'])-1		
			self.cutoff_rigidity_map[i][j] = row['cutoff_rigidity']

		self.cutoff_rigidity_map_T = self.cutoff_rigidity_map.T 
		self.cormap_longitude_centers = (self.cormap_longitude_edges[:-1] + self.cormap_longitude_edges[1:]) / 2.
		self.cormap_latitude_centers = (self.cormap_latitude_edges[:-1] + self.cormap_latitude_edges[1:]) / 2.

	def plot_cutoff_rigidity_map(self,foutname_base='cormap'):

		fig = plt.figure(figsize=(12,8))
		ax = plt.axes(projection=ccrs.PlateCarree())	
		ax.stock_img()
		ax.coastlines(resolution='110m')		

		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)		
		ax.set_xlabel('Longitude (deg)')
		ax.set_ylabel('Latitude (deg)')		
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])	

		X, Y = np.meshgrid(self.cormap_longitude_edges, self.cormap_latitude_edges)
		ax.pcolormesh(X, Y, self.cutoff_rigidity_map_T)

		cormap_contour = ax.contour(
			self.cormap_longitude_centers, 
			self.cormap_latitude_centers, 
			self.cutoff_rigidity_map_T,
			levels=10, colors=['black'],
			transform=ccrs.PlateCarree())
		cormap_contour.clabel(fmt='%1.1f', fontsize=12)
		plt.savefig("%s.pdf" % foutname_base,bbox_inches='tight')

	def get_cutoff_rigidity(self,longitude_deg,latitude_deg):
		i = int(np.digitize(longitude_deg, self.cormap_longitude_edges)) - 1
		j = int(np.digitize(latitude_deg, self.cormap_latitude_edges)) - 1		
		return(self.cutoff_rigidity_map[i][j])

	def set_hvoff_region(self):
		# https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
		open(self.param["lookuptable_hvoff"])



