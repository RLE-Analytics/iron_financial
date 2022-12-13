#!/usr/bin/env python3

from os import system, chdir

chdir('/home/jacobrozran')

from prob_change_functions import create_data, post_to_xano

prob_change = create_data()

post_to_xano(prob_change)

system('sudo shutdown -h now')
