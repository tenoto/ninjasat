# -*- coding: utf-8 -*-

import yaml 

class NinjaSat():
	def __init__(self,setup_yamlfile):
		self.param = yaml.load(open(setup_yamlfile))
		print(self.param)