# -*- coding: utf-8 -*-

import yaml 
import datetime
#import pyproj 

from pyorbital.orbital import Orbital
from pyorbital.tlefile import Tle

#from orbit_predictor.sources import EtcTLESource
#from orbit_predictor.locations import Location

#pyproj_ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
#pyproj_lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
#pyproj_ecef2lla = pyproj.Transformer.from_proj(pyproj_ecef, pyproj_lla) 

#from pymap3d.ecef import ecef2geodetic

class CubeSat():
	def __init__(self,setup_yamlfile):
		self.param = yaml.load(open(setup_yamlfile),Loader=yaml.SafeLoader)

		self.tle = Tle(self.param["tle_name"],self.param["tle_file"])
		self.orbit = Orbital(self.param["tle_name"], line1=self.tle.line1, line2=self.tle.line2)

	def get_lonlatalt(self,utctime):
		lon, lat, alt = self.orbit.get_lonlatalt(utctime)
		return lon, lat, alt
	#	self.source = EtcTLESource(filename=self.param["orbit_tle_parameter"])
	#	self.predictor = self.source.get_predictor("ISS")
	#
	#def get_ecef_location(self,utctime):
	#	return self.predictor.get_position(utctime).position_ecef
	#
	#def get_lla_location(self,utctime):
	#	x,y,z = self.get_ecef_location(utctime)
		#lon,lat,alt =  pyproj_ecef2lla.transform(x,y,z,radians=False)
	#	return ecef2geodetic(x,y,z) # lat, lon, alt
