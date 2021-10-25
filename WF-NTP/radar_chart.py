import numpy
import matplotlib.pyplot as plt
import os,sys


try :
    import csv_dict
except Exception :
    csv_dict=None

plt.rcdefaults()
plt.rc('figure',facecolor='white')

######## COPIED PARAMETERS AND FUNCTION FROM PLOTTER MODULE #############
text_sizes={
'value_labels':18 ,
'xlabels':18 ,
'xlabels_many':'small',
'xlabel': 22 ,
'ylabel': 22 ,
'title' : 24,
'legend_size':18
}

publication={
'value_labels':22 ,
'xlabels':22 ,
'xlabels_many':15,
'xlabel': 30 ,
'ylabel': 30 ,
'title' : 30,
'legend_size':24
}
publication_cm={
'value_labels':6 ,
'xlabels':6 ,
'xlabel': 8 ,
'ylabel': 8 ,
'title' : 8,
'legend_size':8
}
publication_small={
'value_labels':26 ,
'xlabels':30 ,
'xlabels_many':17,
'xlabel': 30 ,
'ylabel': 30 ,
'title' : 30,
'legend_size':30
}

grid_parameters={ # hgrid and vgrid are in default_parameters. these contain only grid styles
'hcolor':'DarkGrey',
'vcolor':'DarkGrey',
'h_ls':':',
'v_ls':':',
'h_lw':0.75,
'v_lw':0.75
}

default_parameters={
'hgrid':True,
'vgrid':True,
'frame':True,
'ticks_nbins':5,
'seq_max_res_per_plot':200,
'markersize':8,
'linewidth':1.5,
'barlinewidth':0.5,
'value_label_text_offset':8,
'all_tight':False,
'dpi':300
}
default_error_bars={
'capsize':4 ,
'capthick':1.,
'elinewidth':1.
}

plt.rc('xtick', labelsize=text_sizes['xlabels'])
plt.rc('ytick', labelsize=text_sizes['xlabels'])


def set_publish(all_same_figure_size=False,thick_ticks=True,axis_tickness=True,small_figure=True,no_grids=True,text_sizes=text_sizes,publication=publication,publication_small=publication_small):
    default_error_bars['capsize']=8
    default_error_bars['capthick']=2
    default_error_bars['elinewidth']=2
    default_parameters['value_label_text_offset']=10
    grid_parameters['h_lw'],grid_parameters['v_lw']=1.25,1.25
    grid_parameters['hcolor'],grid_parameters['vcolor']='black','black'
    if small_figure :
        for k in publication :
            text_sizes[k]=publication_small[k] # with an = or with .copy it does not work
        default_parameters['seq_max_res_per_plot']=50
        default_parameters['markersize']=15
        default_parameters['linewidth']=2.5
    else :
        for k in publication :
            text_sizes[k]=publication[k] # with an = or with .copy it does not work
        default_parameters['seq_max_res_per_plot']=100
        default_parameters['markersize']=12
        default_parameters['linewidth']=2
    plt.rc('xtick', labelsize=text_sizes['xlabels'])
    plt.rc('ytick', labelsize=text_sizes['xlabels'])
    plt.rc('ytick.major', width=1.5,size=6)
    plt.rc('ytick.minor', width=1.,size=3)
    plt.rc('xtick.major', width=1.5,size=6)
    plt.rc('xtick.minor', width=1.,size=3)
    plt.rcParams['lines.linewidth'] = 2
    default_parameters['all_tight']=True
    if no_grids :
        for p in ['frame','vgrid','hgrid'] :
            default_parameters[p]=False
    if thick_ticks!=False :
        plt.rc('ytick.major', width=2.5,size=10)
        plt.rc('ytick.minor', width=2,size=6)
        plt.rc('xtick.major', width=2.5,size=10)
        plt.rc('xtick.minor', width=2,size=6)
    if axis_tickness :
        plt.rc('axes', linewidth=2,edgecolor='black')
    return

class cycle_list(list):
    def __init__(self,l=[]):
        list.__init__(self,l)
    def __getitem__(self,y) :
        #x.__getitem__(y) <==> x[y]
        if type(y) is int and y>=len(self) :
            y=y%len(self)
        return list.__getitem__(self,y)

iworkpalette=cycle_list( [(0.25098039215686274, 0.4470588235294118, 0.792156862745098), (0.4235294117647059, 0.6831372549019608, 0.24313725490196078), (0.8470588235294118, 0.7507843137254902, 0.16784313725490197), (0.8196078431372549, 0.5333333333333333, 0.15294117647058825), (0.7764705882352941, 0.29411764705882354, 0.10980392156862745), (0.4549019607843137, 0.3254901960784314, 0.6509803921568628)] )


################### READ INPUT #########################
def read_results(results_files_folder, grep_section=['Full data set','Cut-off data','Raw data (cut-off)']) :
    data={}
    if type(results_files_folder) is not str : # list of results files
        fl=results_files_folder
    elif os.path.isdir(results_files_folder) :
        if results_files_folder[-1]!='/' : results_files_folder+='/'
        fl= [ results_files_folder+f for f in os.listdir(results_files_folder) if '.'!=f[0] ]
    else : # one single file
        fl=[results_files_folder]
    for fil in fl :
        k=fil.split('/')[-1].split('.')[0] # get fname without path or extension
        if k in data : print "**ERROR** overwriting data for key %s with file %s" % (k,fil)
        data[k] = read_results_file(fil, grep_section=grep_section)
    return data

def read_results_file(fname, grep_section=['Full data set','Cut-off data','Raw data (cut-off)']) :
    data={} # first keys are those in grep_section
    readk=0
    read=False
    for j,line in enumerate(open(fname)) :
      line=line[:-1] # remove \n
      try:
        if len(line)<3 or line[0]=='#': continue
        if line[:5]=='-'*5 :
            if not readk : read=False
            readk= 1-readk # change readkeyword
            continue
        if readk :
            key=line.strip()
            fk=False
            if grep_section == None:
                read = 'Results for' in key
                fk = None
            else:
                for k in grep_section :
                   if key[:len(k)].lower()==k.lower() :
                       if fk!=False : print "ERROR %s found twice (%s)" % (k,key)
                       fk=k
                if fk!=False : read=True
                else : read=False
            continue
        if read :
            if ':' in line : # reading global data
                print line
                if fk not in data : data[fk]={}
                k,val=line.split(':')
                if ',' in val : val=map(float,[v for v in val.split(',') if v.strip()!=''])
                else : val=float(val)

                if k in data[fk] : print "ERROR overwriting %s in %s" % (k,fk)
                data[fk][k]=val
            else : # we assume this is a spreadsheet like data
                if csv_dict==None :
                   print "**ERROR** cannot read data for keyword %s as the csv_dict module was not found. Skipping this section" % (fk)
                   read=False
                   continue
                if ';' in line :
                   line = line.split(';')
                   vals=[]
                   for v in line :
                       try : vals+=[float(v)]
                       except Exception : vals+=[v]
                   if vals[0] in data[fk] : print "ERROR overwriting %s in %s" % (line[0],fk)
                   data[fk][vals[0]]=vals[1:]
                else : # read header
                   if fk in data :  print "ERROR overwriting header and class for section %s" % (fk)
                   hd=line.split('    ')
                   data[fk]=csv_dict.Data()
                   data[fk].key_column_hd_name =hd[0]
                   data[fk].hd= csv_dict.list_to_dictionary(hd[1:])

      except Exception :
         print "\nException raised for file %s at line %d:\n%s" % (fname,j+1,str(line))
         raise

    return data


def get_corners_from_data(data, plot_section='Full data set',  corners=['BPM','Amplitude','Average Speed','Displacement/bend','Survival','Dead ratio'],debug=True) :
    '''
    parse data dictionary and returns specific values (those in corners)
    return plot_values,plot_errors,plot_st_deviations
    '''
    plot_values={}
    plot_errors={}
    plot_st_deviations={}
    for k in data[plot_section] :
        kl=k.lower()
        kl=kl.replace('averaged speed','average speed') # fix possible bug in script that prints results, sometimes it is printed as averageD and sometimes as average
        for c in corners :
            if c.lower() in kl and kl.replace(c.lower(),'').strip() in ['', 'deviation', 'standard deviation','error','mean','error on mean']: # second part makes condition more stringent so that one can distinguish average speed from speed and similar
                if 'deviation' in kl :
                     if debug : print c,'_deviation ->',k
                     if c in plot_st_deviations : print "ERROR overwriting %s in plot_st_deviations" % (c)
                     plot_st_deviations[c]=data[plot_section][k]
                elif 'error' in kl :
                     if debug : print c,'_error ->',k
                     if c in plot_errors : print "ERROR overwriting %s in plot_errors" % (c)
                     plot_errors[c]=data[plot_section][k]
                elif 'mean' in kl :
                     if debug : print c,'_value ->',k
                     if c in plot_values : print "ERROR overwriting %s in plot_values" % (c)
                     plot_values[c]=data[plot_section][k]
    for c in corners :
        if c not in plot_values and  c not in plot_st_deviations and  c not in plot_errors : # maybe this variable does not have mean etc but it is present alone
          tmp=None
          for k in data[plot_section] :
             kl=k.lower()
             kl.replace('averaged speed','average speed') # fix possible bug in script that prints results, sometimes it is printed as averageD and sometimes as average
             if c.lower() in kl : # look for keywords without mean/error/stdev such as the dead ratio
                if tmp==None : tmp=k
                else : print "ERROR %s not found in any with standard criteria, now found twice in %s and %s" % (c,tmp,k)
          if tmp!=None :
              plot_values[c]=data[plot_section][tmp]
              plot_st_deviations[c]=0.
              plot_errors[c]=0. # assign zero
        if c not in plot_values : print "WARNING %s not found for plot_values" % (c)
        if c not in plot_st_deviations : print "WARNING %s not found for plot_st_deviations" %(c)
        if c not in plot_errors :
             if c in plot_st_deviations :
                plot_errors[c]=plot_st_deviations[c]
                print "WARNING %s not found for plot_errors - assinging stdev=%lf" %(c,plot_st_deviations[c])
             else :
                plot_errors[c]=0
                print "WARNING %s not found for plot_errors - assinging 0" %(c)
    return plot_values,plot_errors,plot_st_deviations



#################### WORM-SPECIFIC PLOT ######################
def plot_results( results_files_folder, reference_values, plot_section='Full data set', corners=['BPM','Amplitude','Average Speed','Displacement/bend','Survival','Dead ratio'],ref_label=None, plot_reference_values=True,ngrid=5,ymax=5,figure_size=(8, 8),legend_labels=None,color_palette=None,save=None) :
    '''
    reference_values can be a list of maximum values for the variables given in *corners* (in the same order)
      or a file in the results.txt format from where this values are read. This can be for instance the results for the WT worm
      if None is given then the normalization is done with the maximum values read from the result file(s).
    '''
    print "Going to plot: ",corners
    if reference_values==None : plot_reference_values=False # rest fixed afterwards
    elif type(reference_values) is str : # read file
        reference_data= read_results(reference_values)
        if ref_label==None : ref_label=reference_data.keys()[0]
        plot_values,plot_errors,plot_st_deviations = get_corners_from_data(reference_data.values()[0],corners=corners, plot_section=plot_section)
        ref_values= numpy.array([ plot_values[c] for c in corners ])
        ref_errors= [ plot_errors[c] for c in corners ]
    elif type(reference_values) is list : # use these values
        ref_values=numpy.array(reference_values)
        ref_errors=None
        if ref_label==None : ref_label='Reference'
    if type(plot_section) is str : data= read_results(results_files_folder, grep_section=[plot_section])
    else : data= read_results(results_files_folder, grep_section=plot_section)
    if color_palette is None : color_palette=iworkpalette

    labels=[]
    values=[]
    errors=[]
    for fk in data : # double for loop necessary if reference_values is None
        if fk==ref_label :
            print "WARN %s is ref_label excluding from plotting data.." % (fk)
            continue
        labels+=[fk]
        plot_values,plot_errors,plot_st_deviations = get_corners_from_data(data[fk], plot_section=plot_section,corners=corners)
        values+=[ [ plot_values[c] for c in corners ] ]
        errors +=[ [ plot_errors[c] for c in corners ] ]
    if reference_values==None :
        ref_values=numpy.array(values).max(axis=0)
        ref_errors=None
        if ref_label==None : ref_label='Reference'
    titles=[]
    axlabels=[]
    for j,c in enumerate(corners) :
        titles+=[c]
        axlabs=numpy.linspace(0,ref_values[j],ngrid+1 )[1:] # remove first zero
        s_digits=2
        if max(axlabs)<0.01 : s_digits=4
        elif max(axlabs)<0.1 : s_digits=3
        elif max(axlabs)>10. : s_digits=1
        elif max(axlabs)>100 : s_digits=0
        axlabels+=[ numpy.round(axlabs,s_digits) ] # may need rounding
    fig=plt.figure(figsize=figure_size)
    radar = Radar(fig, titles, axlabels )
    off=0
    if legend_labels!=None : labels=legend_labels
    if plot_reference_values :
        vals= float(ymax)*numpy.array(ref_values)/ ref_values
        if ref_errors!=None and ref_errors!=[None]: yerr= float(ymax)*numpy.array(ref_errors)/ ref_values
        else : yerr=None
        radar.plot(vals,  ls="-", color=  color_palette[off], yerr=yerr, label=ref_label)
        off+=1
    for j, data_set in enumerate(values) :
        vals= float(ymax)*numpy.array(data_set)/ ref_values
        if errors[j]!=None and  errors[j]!=[None]: yerr= float(ymax)*numpy.array(errors[j])/ ref_values
        else : yerr=None
        radar.plot(vals,  ls="-", color=  color_palette[j+off], yerr=yerr, label=labels[j])

    radar.ax.legend(loc='lower left', bbox_to_anchor=(0.85, 0.9),fontsize=text_sizes['legend_size']) # put lower left corner of legend at axis coordinates (frac_x, frac_y), in this way legend can go outside figure..
    plt.draw()
    if save!=None :
        dpi=None
        if '.' not in save : save+='.pdf'
        if dpi==None : dpi=default_parameters['dpi']
        fig.savefig(save, dpi=dpi,bbox_inches="tight",transparent=True) #  bbox_inches=0 remove white space around the figure.. ,
    plt.show()
    return fig





#################### CREATION OF RADARD PLOT ##################
class Radar(object):
    def __init__(self, fig, titles, labels, rect=None,debug=False,transparent=True,rotation_displacement=15,ngrid=5,ymax=5):
        '''
         this allow to put multiple axis (hence radard) in same figure by giving different rect each time
         *rect* [*left*, *bottom*, *width*, *height*] where all quantities are in fractions of figure width and height.
        '''
        if rect is None:
            rect = [0.1, 0.05, 0.8, 0.9]
        self.n = len(titles)
        self.angles = numpy.linspace(rotation_displacement, 360+rotation_displacement, self.n+1)[:-1] # last will be 360=0 so we put +1
        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i)
                         for i in range(self.n)]

	if debug : print self.n,len(self.axes),len(self.angles),len(labels),len(titles),titles,self.angles
        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=titles, fontsize=text_sizes['title'],frac=1.15)
        #*frac* is the fraction of the polar axes radius at which to place the label (1 is the edge). e.g., 1.05 is outside the axes and 0.95 is inside the axes. 1.15 is 15% out

        if transparent : self.ax.patch.set_visible(False)# MAKES IT TRANSPARENT
        for ax in self.axes[1:] : # note that 0 is self.ax above, here we remove the others
            ax.patch.set_visible(False)
            ax.grid("off") # removes ugly grid (put back after with set_rgrid)
            ax.xaxis.set_visible(False) # removes polar angle labels and ticks corresponding to x-axis (useless in radard plot)


        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids( numpy.linspace(1,ymax,ngrid), angle=angle, labels=label,ha='center',va='center')
            ax.spines["polar"].set_visible(False) # remove outer circle (e.g. frame)
            ax.set_ylim(0, ymax)

    def plot(self, values,yerr=None,yerr_ls=None,fill_errors=True, *args, **kw):
        if yerr_ls==None : # set to default
          if fill_errors : yerr_ls=''
          else : yerr_ls='--'
        angle = numpy.deg2rad(numpy.r_[self.angles, self.angles[0]])
        values=numpy.array(values)
        vals = numpy.r_[values, values[0]]
        self.ax.plot(angle, vals, *args, **kw)
        if yerr!=None :
           if len(yerr)==2 or ( hasattr(yerr[0],'__len__') and len(yerr[0])==2 ): # we gave asymmetric error bars, e.g. Confidence Interval
                if ( hasattr(yerr[0],'__len__') and len(yerr[0])==2 ): yerr=zip(*yerr)
                vvL, vvU = values - numpy.array(yerr[0]) , values + numpy.array(yerr[1])
           else :
                vvL= values - numpy.array(yerr)
                vvU=values + numpy.array(yerr)
           vvL  = numpy.r_[vvL, vvL[0]]
           vvU = numpy.r_[vvU, vvU[0]]
           if 'ls' in kw : del kw['ls']
           if 'label' in kw : del kw['label']
           self.ax.plot(angle, vvL,ls=yerr_ls, *args, **kw)
           self.ax.plot(angle, vvU,ls=yerr_ls, *args, **kw)
           if 'color' in kw : facecolor=kw['color']
           else : facecolor='DimGray'
           if fill_errors : self.ax.fill_between(angle,vvL,vvU,   where=vvU >= vvL, facecolor=facecolor,alpha=0.3, interpolate=True,edgecolor='none')



if __name__ == '__main__': # run example
    if os.path.exists('worm_ab_results') :
        plot_results('worm_ab_results',None,save='worm_radard_charts.png') # 'worm_ab_results/results_DAY7_N2_NEGATIVE.txt' has amplitude smaller than others, also obviously dead-to-live ratio is smaller
        exit()

    fig = plt.figure(figsize=(8, 8))

    set_publish()
    titles = list("ABCDEF")

    labels = [
        list("abcde"), list("12345"), list("uvwxy"),
        ["one", "two", "three", "four", "five"],
        list("jklmn"),range(10,16)
    ]

    radar = Radar(fig, titles, labels, debug=True)
    radar.plot([1, 3, 2, 5, 4,3],  ls="-", color="b", alpha=0.4, label="first")
    radar.plot([2.3, 2, 3, 3, 2,5],ls="-", color="r", alpha=0.4, label="second")
    radar.plot([3, 4, 3, 4, 2,1], ls="-", color="g", alpha=1,yerr=[0.1,0.1,0.31,0.11,0.1,0.2], label="third")
    radar.ax.legend(loc='lower left', bbox_to_anchor=(0.85, 0.9)) # put lower left corner of legend at axis coordinates (frac_x, frac_y), in this way legend can go outside figure..
    plt.tight_layout()
    plt.draw()
    plt.show()




