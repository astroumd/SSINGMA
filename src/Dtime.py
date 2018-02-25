#! /usr/bin/env python
#

from __future__ import print_function

import time
import numpy as np
import logging

#  For some variations on this theme, e.g.  time.time vs. time.clock, see
#  http://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python

ostype = None

class DtimeSimple(object):
    """ Class to help measuring the wall clock time between tagged events
        Typical usage:
    
        dt = Dtime('some_label')
        ...
        dt.tag('a')
        ...
        dt.tag('b')
        dt.end()
    """
    def __init__(self, label=".", report=True):
        self.start = self.time()
        self.init = self.start
        self.label = label
        self.report = report
        self.dtimes = []
        dt = self.init - self.init
        if self.report:
            #logging.info("Dtime: %s ADMIT " % (self.label + self.start)) #here took out a '
            logging.info("Dtime: %s BEGIN " % self.label + str(dt))

    def reset(self, report=True):
        self.start = self.time()
        self.report = report
        self.dtimes = []

    def tag(self, mytag):
        t0 = self.start
        t1 = self.time()
        dt = t1 - t0
        self.dtimes.append((mytag, dt))
        self.start = t1
        if self.report:
            logging.info("Dtime: %s " % self.label + mytag + "  " + str(dt))
        return dt

    def show(self):
        if self.report:
            for r in self.dtimes:
                logging.info("Dtime: %s " % self.label + str(r[0]) + "  " + str(r[1]))
        return self.dtimes

    def end(self):
        t0 = self.init
        t1 = self.time()
        dt = t1 - t0
        if self.report:
            logging.info("Dtime: %s END " % self.label + str(dt))
        return dt

    def time(self):
        """ pick the actual OS routine that returns some kind of timer
        time.time   :    wall clock time (include I/O and multitasking overhead)
        time.clock  :    cpu clock time
        """
        return np.array([time.clock(), time.time()])

    def get_mem(self):
        """ Read memory usage info from /proc/pid/status
            Return Virtual and Resident memory size in MBytes.

            NOTE: not implemented here, see the ADMIT version if you need this.
        """
        return np.array([])       # NA yet


    
class Dtime(object):
    """ Class to help measuring the wall clock time between tagged events
        Typical usage:
        dt = Dtime()
        ...
        dt.tag('a')
        ...
        dt.tag('b')
    """
    def __init__(self, label=".", report=True):
        self.start = self.time()
        self.init = self.start
        self.label = label
        self.report = report
        self.dtimes = []
        dt = self.init - self.init
        if self.report:
            # logging.timing("%s ADMIT " % self.label + str(self.start))
            logging.info("%s BEGIN " % self.label + str(dt))
            # logging.info("Dtime: %s BEGIN " % self.label + str(dt))            

    def reset(self, report=True):
        self.start = self.time()
        self.report = report
        self.dtimes = []

    def tag(self, mytag):

        t0 = self.start
        t1 = self.time()
        dt = t1 - t0

        # get memory usage (Virtual and Resident) info
        mem = self.get_mem()
        if mem.size != 0 :
            dt = np.append(dt, mem)

        self.dtimes.append((mytag, dt))
        self.start = t1
        if self.report:
            logging.info("%s " % self.label + mytag + "  " + str(dt))
        return dt

    def show(self):
        if self.report:
            for r in self.dtimes:
                logging.info("%s " % self.label + str(r[0]) + "  " + str(r[1]))
        return self.dtimes

    def end(self):
        t0 = self.init
        t1 = self.time()
        dt = t1 - t0
        if self.report:
            logging.info("%s END " % self.label + str(dt))
        return dt

    def time(self):
        """ pick the actual OS routine that returns some kind of timer
        time.time   :    wall clock time (include I/O and multitasking overhead)
        time.clock  :    cpu clock time
        """
        return np.array([time.clock(), time.time()])

    def get_mem(self):
        """ Read memory usage info from /proc/pid/status
            Return Virtual and Resident memory size in MBytes.
        """
        global ostype
        
        if ostype == None:
            ostype = os.uname()[0].lower()
            logging.info("OSTYPE: %s" % ostype)
            
        scale = {'MB': 1024.0}
        lines = []
        try:
            if ostype == 'linux':
                proc_status = '/proc/%d/status' % os.getpid()          # linux only
                # open pseudo file  /proc/<pid>/status
                t = open(proc_status)
                # get value from line e.g. 'VmRSS:  9999  kB\n'
                for it in t.readlines():
                    if 'VmSize' in it or 'VmRSS' in it :
                        lines.append(it)
                t.close()
            else:
                proc = subprocess.Popen(['ps','-o', 'rss', '-o', 'vsz', '-o','pid', '-p',str(os.getpid())],stdout=subprocess.PIPE)
                proc_output = proc.communicate()[0].split('\n') 
                proc_output_memory = proc_output[1]
                proc_output_memory = proc_output_memory.split()
                
                phys_mem = int(proc_output_memory[0])/1204 # to MB 
                virtual_mem = int(proc_output_memory[1])/1024 
                
        except (IOError, OSError):
            if self.report:
                logging.info(self.label + " Error: cannot read memory usage information.")

            return np.array([])

        # parse the two lines
    
        mem = {}
        if(ostype != 'darwin'):
            for line in lines:
                words = line.strip().split()
            #print words[0], '===', words[1], '===', words[2]
                
            # get rid of the tailing ':'
                key = words[0][:-1]

            # convert from KB to MB
                scaled = float(words[1]) / scale['MB']
                mem[key] = scaled
        else:
            mem['VmSize'] = virtual_mem
            mem['VmRSS']  = phys_mem


        return np.array([mem['VmSize'], mem['VmRSS']])



#if __name__ == '__main__':
if False:
    logging.basicConfig(level = logging.INFO)
    dt = Dtime("testingDtime")
    dt.tag('one')
    # print("Hello Dtime World")
    print("Hello Dtime World")
    dt.tag('two')
    dt.end()
       


    
