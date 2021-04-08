# -*- coding: utf-8 -*-
import newColumnTracer

c= newColumnTracer.ColumnTracer()
xs = [0,.2,.4,.6,.8,1]
ts = [0,1,2,3,4,5]
output = c.effluent_concentration(time_end=10, interval=0.1, plot = True, print_conc=True)
