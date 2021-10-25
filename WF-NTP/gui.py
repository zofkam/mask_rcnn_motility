import Tkinter as tk
import ttk
import tkFileDialog
import tkMessageBox
import tkSimpleDialog
import cv2
import cv2.cv as cv
import os
import multiprocessing
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import key_press_handler
from copy import copy, deepcopy
from functools import partial
import matplotlib.pyplot as plt
from plot_path import plot_path
import radar_chart

def run_tracker(tup):
    filename, index = tup
    print(filename)
    os.system('python multiwormtracker.new.py %s'%filename)
    return filename, index

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.protocol('WM_DELETE_WINDOW', self.exit)
        self.jobindex = 0
        self.jobs = {}
        self.job_buttons = {}
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count())
        self.editing_job = False

        # Main window
        parent.wm_title("Multiwormtracker")
        sizex = 800 if os.name=='nt' else 1000
        sizey = 600
        posx  = 100
        posy  = 100
        logheight = 200
        consoleheight = 310
        parent.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))
        parent.resizable(0,0)

        # Overview
        logframe = tk.Frame(parent,relief=tk.GROOVE,
                width=sizex-50,height=logheight,bd=1)
        logframe.place(x=10,y=10)

        canvas = tk.Canvas(logframe)
        self.logframe = ttk.Frame(canvas)

        logscrollbar = ttk.Scrollbar(logframe,orient="vertical",
                                    command=canvas.yview)
        canvas.configure(yscrollcommand=logscrollbar.set)

        logscrollbar.pack(side="right",fill="y")
        canvas.pack(side="left")
        canvas.create_window((0,0),window=self.logframe,anchor='nw')
        def rescale_logscrollbar(event):
            canvas.configure(scrollregion=canvas.bbox("all"),width=sizex-50,height=logheight)
        self.logframe.bind("<Configure>",rescale_logscrollbar)
        canvas.configure(scrollregion=canvas.bbox("all"),
                         width=sizex-50,height=logheight)

        separator = tk.Frame(self.logframe,width=200,
                        height=1).grid(row=0, column=0)
        separator = tk.Frame(self.logframe,width=400,
                        height=1).grid(row=0, column=1)

        # Console
        consoleframe = tk.Frame(parent,relief=tk.GROOVE,
                width=sizex-50,height=consoleheight,bd=1)
        consoleframe.place(x=10,y=logheight+20)
        canvas2 = tk.Canvas(consoleframe)
        self.consoleframe = ttk.Frame(canvas2)

        self.console = tk.Text(consoleframe, width=93, height=19)
        self.console.place(x=0, y=0)

        # Buttons
        btnframe = tk.Frame(parent, width=sizex,height=50,bd=1)
        btnframe.place(x=10,y=logheight+consoleheight+50)
        canvas3 = tk.Canvas(btnframe)
        self.btnframe = ttk.Frame(canvas3)

        start_btn = ttk.Button(btnframe,text="Start", width=22,
                        command=self.start_all)
        start_btn.pack(padx=5, side="left")
        addjob_btn = ttk.Button(btnframe,text="Add job", width=22,
                        command=self.add_job_dialog)
        addjob_btn.pack(padx=5, side="left")
        load_btn = ttk.Button(btnframe,text="Load job", width=22,
                        command=self.load_job)
        load_btn.pack(padx=5, side="left")
        load_btn = ttk.Button(btnframe,text="Utilities", width=22,
                        command=self.utils)
        load_btn.pack(padx=5, side="left")
        exit_btn = ttk.Button(btnframe,text="Exit", width=22,
                        command=self.exit)
        exit_btn.pack(padx=5, side="left")

    def utils(self):
        Utils(self)

    def start_all(self):
        for key in self.jobs:
            self.start_job(key)

    def make_settings_file(self, job, example=False):
        settings = ''
        settings += '### Input\n'
        settings += 'filename = "%s"\n'%job['video']
        settings += 'start_frame = %s\n'%job['startframe']
        settings += 'limit_images_to = %s\n'%job['useframes']
        settings += 'fps = %s\n'%job['fps']
        settings += 'px_to_mm = %s\n'%job['px_to_mm']
        settings += 'darkfield = %s\n'%job['darkfield']
        if example:
            settings += 'stop_after_example_output = True\n'
        else:
            settings += 'stop_after_example_output = False\n'
        settings += '\n### Output\n'
        settings += 'save_as = "%s"\n'%(job['outputname'].rstrip('/')+'/')
        settings += 'output_overlayed_images = %s\n'%job['outputframes']
        settings += 'font_size =  %s\n'%job['font_size']
        settings += 'fig_size = (20,20)\n' # currently not changeable
        settings += 'scale_bar_size = 1.0\n' # currently not changeable
        settings += 'scale_bar_thickness = 7\n' # currently not changeable
        settings += '\n### Z-filtering\n'
        settings += 'use_images = %s\n'%job['z_use']
        settings += 'use_around = %s\n'%job['z_padding']
        settings += 'Z_skip_images = 1\n'
        settings += '\n### Thresholding\n'
        settings += 'keep_dead_method = %s\n'%(job['method']=='Keep Dead')
        settings += 'std_px = %s\n'%job['std_px']
        settings += 'threshold = %s\n'%job['threshold']
        settings += 'opening = %s\n'%job['opening']
        settings += 'closing = %s\n'%job['closing']
        settings += 'skeletonize = %s\n'%job['skeletonize']
        settings += 'prune_size = %s\n'%job['prune_size']
        settings += 'do_full_prune = %s\n'%job['do_full_prune']
        settings += '\n### Locating\n'
        settings += 'min_size = %s\n'%job['minsize']
        settings += 'max_size = %s\n'%job['maxsize']
        settings += 'minimum_ecc = %s\n'%job['minimum_ecc']
        settings += '\n### Form trajectories\n'
        settings += 'max_dist_move = %s\n'%job['maxdist']
        settings += 'min_track_length = %s\n'%job['minlength']
        settings += 'memory = %s\n'%job['memory']
        settings += '\n### Bending statistics\n'
        settings += 'bend_threshold = %s\n'%job['bendthres']
        settings += 'minimum_bends = %s\n'%job['minbends']
        settings += '\n### Velocities\n'
        settings += 'frames_to_estimate_velocity = %s\n'%job['velframes']
        settings += '\n### Dead worm statistics\n'
        settings += 'maximum_bpm = %s\n'%job['maxbpm']
        settings += 'maximum_velocity = %s\n'%job['maxvel']
        settings += '\n### Regions\n'
        settings += 'regions = %s\n'%job['regions']
        settings += '\n### Optimisation tools\n'
        settings += 'lower = %s\n'%job['lower'] #added new
        settings += 'upper = %s\n'%job['upper'] #added new
        settings += 'use_average = %s\n'%(job['use_average']=='Average') #added
        settings += 'cutoff_filter = %s\n'%job['cutoff_filter']
        settings += 'extra_filter = %s\n'%job['extra_filter'] # added new
        settings += 'Bends_max = %s\n'%job['Bends_max']  # added new
        settings += 'Speed_max = %s\n'%job['Speed_max']  # added new
        return settings

    def start_job(self, index):
        job = self.jobs[index]
        job_buttons = self.job_buttons[index]
        job_buttons['progressbar'].start()

        # Make output directory
        try:
            os.mkdir(job['outputname'])
        except OSError:
            self.log('Warning: job folder "%s" already created, overwriting.'%job['outputname'])

        settings = self.make_settings_file(job)
        settingsfilename = job['outputname'].rstrip('/')+'/settings.py'
        with open(settingsfilename,'w') as f:
            f.write(settings)

        self.log('Job: "%s" started.'
                    %job['video'].split('/')[-1])
        self.pool.apply_async(run_tracker,((settingsfilename,index),),
                callback=self.finished)

    def finished(self,tup):
        filename, index = tup
        self.log('Finished '+filename)
        self.job_buttons[index]['progressbar'].stop()

    def log(self, txt):
        self.console.config(state=tk.NORMAL)
        if txt[-1]!='\n':
            txt += '\n'
        self.console.insert(tk.END, txt)
        self.console.see('end')
        self.console.config(state=tk.DISABLED)

    def add_job_dialog(self):
        add_job_dialog = AddJob(self)

    def add_job(self, job):
        videonames = job['video'].split(', ')
        for videoname in videonames:
            i = self.jobindex
            this_job = deepcopy(job)
            this_job['video'] = videoname

            short_videoname = videoname.split('/')[-1]
            append_name = ".".join(short_videoname.split('.')[:-1])

            if len(videonames)>1:
                this_job['outputname'] += '_'+append_name

            self.jobs[i] = this_job

            if len(short_videoname)>25:
                short_videoname = short_videoname[:25]
            short_outputname = this_job['outputname']
            if len(short_outputname)>50:
                short_outputname = short_outputname[-50:]

            thisframe = ttk.Frame(self.logframe, width=300, height=300)
            thisframe.grid(row=i, column=0,sticky='w')

            deletebtn = ttk.Button(thisframe,text='X',width=2,
                                    command=partial(self.delete_job,i))
            deletebtn.pack(side='left')
            videobtn = ttk.Button(thisframe,text=short_videoname, width=30,
                                    command=partial(self.edit_job,i))
            videobtn.pack(side='left')

            examplebtn = ttk.Button(thisframe,text="Example", width=9,
                                    command=partial(self.example_output,i))
            examplebtn.pack(side='left')

            jobinfo = ttk.Label(thisframe,text=short_outputname)
            jobinfo.pack(side='left')
            progressbar = ttk.Progressbar(thisframe, orient='horizontal',
                length=140, mode='indeterminate')
            progressbar.pack(side='right')

            self.job_buttons[i] = {'videobtn':videobtn,
                                   'jobinfo':jobinfo,
                                    'progressbar': progressbar,
                                    'deletebtn':deletebtn,
                                    'thisframe':thisframe}
            if not self.editing_job:
                self.log('Job: "%s" successfully added.'
                            %videoname.split('/')[-1])
            self.jobindex += 1
            self.editing_job = False

    def edit_job(self, index):
        self.editing_job = True
        self.editing_index = index
        AddJob(self)

    def load_job(self):
        filename = tkFileDialog.askopenfilename(title='Locate a settings.py file',filetypes=[("Settings file","*.py")])
        try:
            with open(filename) as f:
                settings = f.read()
            exec settings
        except:
            self.log('Not a valid settings.py file')
            return
        try:
            job = {}
            job['video'] = filename
            job['startframe'] = start_frame
            job['useframes'] = limit_images_to
            job['fps'] = fps
            job['px_to_mm'] = px_to_mm
            job['darkfield'] = darkfield
            job['method'] = 'Keep Dead' if keep_dead_method else 'Z-Filtering'
            job['z_use'] = use_images
            job['z_padding'] = use_around
            job['std_px'] = std_px
            job['threshold'] = threshold
            job['opening'] = opening
            job['closing'] = closing
            job['minsize'] = min_size
            job['maxsize'] = max_size
            job['maxdist'] = max_dist_move
            job['minlength'] = min_track_length
            job['memory'] = memory
            job['bendthres'] = bend_threshold
            job['minbends'] = minimum_bends
            job['velframes'] = frames_to_estimate_velocity
            job['maxbpm'] = maximum_bpm
            job['maxvel'] = maximum_velocity
            job['outputname'] = save_as
            job['outputframes'] = output_overlayed_images
            job['font_size'] = font_size
            job['extra_filter'] = extra_filter # added new
            job['cutoff_filter'] = cutoff_filter #new
            job['lower'] = lower # added new
            job['upper'] = upper # added new
            job['use_average'] = 'Average' if use_average else 'Maximum' # added new
            job['Bends_max'] = Bends_max # added new
            job['Speed_max'] = Speed_max # added new
            try: # Backwards compability
                job['minimum_ecc'] = minimum_ecc
            except NameError:
                job['minimum_ecc'] = 0.0
            try: # Backwards compability
                job['skeletonize'] = skeletonize
                job['prune_size'] = prune_size
            except NameError:
                job['skeletonize'] = False
                job['prune_size'] = 0
            try: # Backwards compability
                job['do_full_prune'] = do_full_prune
            except NameError:
                job['do_full_prune'] = False
            job['regions'] = regions
        except:
            self.log('Not a valid settings.py file')
            return
        self.add_job(job)

    def example_output(self, index):
        job = self.jobs[index]
        job_buttons = self.job_buttons[index]
        job_buttons['progressbar'].start()

        # Make output directory
        try:
            os.mkdir(job['outputname'])
        except OSError:
            self.log('Warning: job folder "%s" already created, overwriting.'%job['outputname'])

        settings = self.make_settings_file(job, example=True)
        settingsfilename = job['outputname'].rstrip('/')+'/settings.py'
        with open(settingsfilename,'w') as f:
            f.write(settings)

        self.log('Job: "%s" example output started.'
                    %job['video'].split('/')[-1])
        self.pool.apply_async(run_tracker,((settingsfilename,index),),
                callback=self.finished_example)

    def finished_example(self,tup):
        filename, index = tup
        self.log('Finished example output '+filename)
        self.job_buttons[index]['progressbar'].stop()

    def delete_job(self, index):
        name = self.jobs[index]['video'].split('/')[-1]
        if tkMessageBox.askokcancel("Delete "+name,
                    "Are you sure you wish delete this job?"):
            self.job_buttons[index]['thisframe'].grid_forget()
            self.log('Deleted job "%s".'%name)
            del self.job_buttons[index]
            del self.jobs[index]

    def exit(self):
        self.pool.close()
        self.parent.destroy()

class Utils(tk.Toplevel):
    def __init__(self, parent,  *args, **kwargs):
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.wm_title("Utilities")
        self.grab_set()
        self.parent = parent
        sizex = 230 if os.name=='nt' else 300
        sizey = 140
        posx  = 200
        posy  = 200
        self.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))
        self.resizable(0,0)

        ttk.Button(self,text="Plot path", width=30,
                    command=self.plotpath).grid(row=0, padx=20, pady=10)

        ttk.Button(self,text="Export to tsv", width=30,
                    command=self.tsv).grid(row=1, padx=20, pady=10)

    def plotpath(self):
        filename = tkFileDialog.askopenfilename(title='Locate a track.p file',filetypes=[("track.p file","*.p")])
        if filename:
            plot_path(filename)

    def tsv(self):
        ToTsv(self)

class ToTsv(tk.Toplevel):
    def __init__(self, parent, *args, **kwargs):
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.parent = parent
        self.main = parent.parent
        # Main window
        self.wm_title("Export tab-seperated file")
        self.grab_set()
        sizex = 500 if os.name=='nt' else 700
        sizey = 400
        posx  = 100
        posy  = 100
        logheight = 335
        consoleheight = 310
        self.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))
        self.resizable(0,0)
        self.index = 0
        self.filenames = []

        # Overview
        logframe = tk.Frame(self,relief=tk.GROOVE,
                width=sizex-50,height=logheight,bd=1)
        logframe.place(x=10,y=10)

        canvas = tk.Canvas(logframe)
        self.logframe = ttk.Frame(canvas)

        logscrollbar = ttk.Scrollbar(logframe,orient="vertical",
                                    command=canvas.yview)
        canvas.configure(yscrollcommand=logscrollbar.set)

        logscrollbar.pack(side="right",fill="y")
        canvas.pack(side="left")
        canvas.create_window((0,0),window=self.logframe,anchor='nw')
        def rescale_logscrollbar(event):
            canvas.configure(scrollregion=canvas.bbox("all"),width=sizex-50,height=logheight)
        self.logframe.bind("<Configure>",rescale_logscrollbar)
        canvas.configure(scrollregion=canvas.bbox("all"),
                         width=sizex-50,height=logheight)

        btnframe = tk.Frame(self, width=sizex,height=50,bd=1)
        btnframe.place(x=10,y=logheight+20)

        ttk.Button(btnframe, text='Add files', width=17,
                command=self.add).grid(row=0,column=0, pady=5)
        ttk.Button(btnframe, text='Add folder', width=17,
                command=self.addrecursive).grid(row=0,column=1, padx=5, pady=5)
        ttk.Button(btnframe, text='Export', width=17,
                command=self.export).grid(row=0,column=2, padx=0, pady=5)
        cancel_btn = ttk.Button(btnframe, width=17, text="Close",
                command=self.destroy).grid(row=0,column=3, padx=5, pady=5)

    def add_filesnames(self, filenames):
        for f in filenames:
            self.filenames.append(f)
            thisframe = ttk.Frame(self.logframe, width=300, height=300)
            thisframe.grid(row=self.index, column=0,sticky='w')
            ttk.Label(thisframe,text=f[-70:]).pack()
            self.index += 1

    def add(self):
        filenames = tkFileDialog.askopenfilenames(title='Locate results.txt files',filetypes=[("results.txt files","*.txt")])
        self.add_filesnames(filenames)

    def addrecursive(self):
        folder = tkFileDialog.askdirectory()
        if folder and tkMessageBox.askquestion("Warning",
                "Recursive searching might take a while for large folders.\n"
                "Are you sure you wish to search this folder?",
                icon='warning'):
            filenames = []
            for f in os.walk(folder):
                if 'results.txt' in f[2]:
                    filenames.append((f[0].rstrip('/\\') + '/results.txt').replace('\\','/'))
            self.add_filesnames(filenames)

    def export(self):
        fnames = self.filenames

        output = []
        first = True
        legends = ['Saved as', 'Movie', 'Region']

        for fname in fnames:
            sep = '---------------------------------'

            save_as = "/".join(fname.split('/')[:-1])
            with open(fname) as f:
                s = f.read()
            if sep not in s:
                self.main.log(fname+' not a results.txt file.')
                continue

            regions = 'Stats for region:'
            skip_first = regions in s

            l = s.split(sep)
            parse_next = False
            for section in l:
                if parse_next:
                    pars = [save_as, moviename, region_name]
                    lines = section.split('\n')
                    for line in lines:
                        if ':' in line:
                            s, n = [x.strip() for x in line.split(':')]
                            if first:
                                legends.append(s)
                            pars.append(n)
                    output.append(pars)

                    parse_next = False
                    first = len(legends)==3

                elif 'Results for' in section:
                    parse_next = not skip_first
                    moviename = section.split('Results for')[-1].strip()
                    region_name = 'all'

                elif 'Stats for region:' in section:
                    parse_next = True
                    region_name = section.split('Stats for region:')[-1]\
                            .strip()

        if len(output)>0:
            out = ''
            for i in range(len(legends)):
                out += legends[i]
                for j in xrange(len(output)):
                    if len(output[j])>i:
                        out += '\t' + output[j][i]
                    elif len(output[j])==i:
                        out += '\t' + '0'
                    else:
                        out += '\t' + 'n/a'
                out += '\n'




            save_fname = tkFileDialog.asksaveasfilename(filetypes=
                        [('*.tsv','Tab seperated file')])
            if save_fname:
                if save_fname[-4:]!='.tsv':
                    save_fname += '.tsv'
                with open(save_fname,'w') as f:
                    f.write(out)
                    
                self.main.log(save_fname+' written.')


class Fingerprint(tk.Toplevel):
    def __init__(self, parent, *args, **kwargs):
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.parent = parent
        self.main = parent.parent
        # Main window
        self.wm_title("Generate worm fingerprints")
        self.grab_set()
        sizex = 500 if os.name=='nt' else 700
        sizey = 400
        posx  = 100
        posy  = 100
        logheight = 335
        consoleheight = 310
        self.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))
        self.resizable(0,0)
        self.index = 0
        self.filenames = []

        # Overview
        logframe = tk.Frame(self,relief=tk.GROOVE,
                width=sizex-50,height=logheight,bd=1)
        logframe.place(x=10,y=10)

        canvas = tk.Canvas(logframe)
        self.logframe = ttk.Frame(canvas)

        logscrollbar = ttk.Scrollbar(logframe,orient="vertical",
                                    command=canvas.yview)
        canvas.configure(yscrollcommand=logscrollbar.set)

        logscrollbar.pack(side="right",fill="y")
        canvas.pack(side="left")
        canvas.create_window((0,0),window=self.logframe,anchor='nw')
        def rescale_logscrollbar(event):
            canvas.configure(scrollregion=canvas.bbox("all"),width=sizex-50,height=logheight)
        self.logframe.bind("<Configure>",rescale_logscrollbar)
        canvas.configure(scrollregion=canvas.bbox("all"),
                         width=sizex-50,height=logheight)

        btnframe = tk.Frame(self, width=sizex,height=50,bd=1)
        btnframe.place(x=10,y=logheight+20)

        ttk.Button(btnframe, text='Add files', width=17,
                command=self.add).grid(row=0,column=0, pady=5)
        ttk.Button(btnframe, text='Make graph', width=17,
                command=self.export).grid(row=0,column=2, padx=5, pady=5)
        cancel_btn = ttk.Button(btnframe, width=17, text="Close",
                command=self.destroy).grid(row=0,column=3, padx=0, pady=5)

    def add_filesnames(self, filenames):
        for f in filenames:
            self.filenames.append(f)
            thisframe = ttk.Frame(self.logframe, width=300, height=300)
            thisframe.grid(row=self.index, column=0,sticky='w')
            ttk.Label(thisframe,text=f[-70:]).pack()
            self.index += 1

    def add(self):
        filenames = tkFileDialog.askopenfilenames(title='Locate results.txt files',filetypes=[("results.txt files","*.txt")])
        self.add_filesnames(filenames)

    def export(self):
        input_files = self.filenames
        section_to_plot = None
        variables_to_plot = ['BPM','Speed','Area','Bend Measure']
        color = None
        legend_labels = ["/".join(f.split('/')[-2:]) for f in input_files]
        save = None
        small_figure = False

        if small_figure : radar_chart.set_publish() # update parameters
        radar_chart.plot_results(input_files, None ,plot_section=section_to_plot ,corners=variables_to_plot,color_palette=color, legend_labels=legend_labels,save='test.png')


class AddJob(tk.Toplevel):
    def __init__(self, parent,  *args, **kwargs):
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.wm_title("Add job")
        self.grab_set()
        self.parent = parent
        sizex = 900 if os.name == 'nt' else 770
        sizey = 950
        posx  = 100
        posy  = 50
        self.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))
        self.resizable(0,0)
        padframes = 7
        self.regions = {}
        self.editing = parent.editing_job
        if self.editing:
            self.edit_index = parent.editing_index
            self.edit_job = parent.jobs[self.edit_index]
            edit_job = self.edit_job
        editing = self.editing

        ###### VIDEO SECTION #######

        videoframe = ttk.LabelFrame(self, text="Video")
        videoframe.grid(row=0, sticky='w')

        videonameframe = ttk.Frame(videoframe)
        videonameframe.grid(row=0)
        self.videoname = ttk.Entry(videonameframe, width=80, state='readonly')

        self.videoname.grid(row=0, padx=10)
        video_btn = ttk.Button(videonameframe, text="Browse",
                    command=self.find_video)
        video_btn.grid(row=0, column=1, padx=5)
        self.videoinfo = ttk.Label(videonameframe, text='No video chosen.')
        self.videoinfo.grid(row=1, sticky='w', padx=10)

        videocutframe = ttk.Frame(videoframe)
        videocutframe.grid(row=1, sticky='w')

        start_frame_label = ttk.LabelFrame(videocutframe, text="Start frame")
        start_frame_label.grid(row=2, sticky='w', padx=10)
        self.start_frame = ttk.Entry(start_frame_label, width=15)
        self.start_frame.insert(0, '0')
        self.start_frame.pack(side='left', padx=3)

        use_frame_label = ttk.LabelFrame(videocutframe, text="Use frames")
        use_frame_label.grid(row=2, column=1, sticky='w', padx=10)
        self.use_frame = ttk.Entry(use_frame_label, width=15)
        self.use_frame.pack(side='left', padx=3)

        fps_label = ttk.LabelFrame(videocutframe, text="FPS")
        fps_label.grid(row=2, column=2, sticky='w', padx=10)
        self.fps = ttk.Entry(fps_label, width=15)
        self.fps.insert(0, '20')
        self.fps.pack(side='left', padx=3)

        px_to_mm_label = ttk.LabelFrame(videocutframe, text="px to mm factor")
        px_to_mm_label.grid(row=2, column=3, sticky='w', padx=10)
        self.px_to_mm = ttk.Entry(px_to_mm_label, width=15)
        self.px_to_mm.insert(0, '0.040')
        self.px_to_mm.pack(side='left', padx=3)

        darkfield_label = ttk.LabelFrame(videocutframe, text="Darkfield")
        darkfield_label.grid(row=2, column=4, sticky='w', padx=10)
        self.darkfield = tk.IntVar()
        self.darkfield_box = tk.Checkbutton(darkfield_label,
                                    variable=self.darkfield)
        self.darkfield_box.pack()

        ###########################

        ###### LOCATING SECTION #######
        locate_frame = ttk.LabelFrame(self, text="Locating")
        locate_frame.grid(row=1, sticky='w', pady=padframes)

        methodframe = ttk.Frame(locate_frame)
        methodframe.grid(row=0)

        method_label = ttk.LabelFrame(methodframe, text="Method")
        method_label.grid(row=0, column=0, sticky='w', padx=10)
        self.method = ttk.Combobox(method_label, values=('Keep Dead','Z-filtering'), state='readonly')
        self.method.pack(side='left', padx=3)
        self.method.current(0)

        z_use_label = ttk.LabelFrame(methodframe, text="Z use images")
        z_use_label.grid(row=0, column=1, sticky='w', padx=10)
        self.z_use = ttk.Entry(z_use_label, width=15)
        self.z_use.insert(0, '100')
        self.z_use.configure(state="disabled")
        self.z_use.pack(side='left', padx=3)

        z_padding_label = ttk.LabelFrame(methodframe, text="Z padding")
        z_padding_label.grid(row=0, column=2, sticky='w', padx=10)
        self.z_padding = ttk.Entry(z_padding_label, width=15)
        self.z_padding.insert(0, '5')
        self.z_padding.configure(state="disabled")
        self.z_padding.pack(side='left', padx=3)

        std_px_label = ttk.LabelFrame(methodframe, text="Std pixels")
        std_px_label.grid(row=0, column=3, sticky='w', padx=10)
        self.std_px = ttk.Entry(std_px_label, width=15)
        self.std_px.insert(0, '64')
        self.std_px.pack(side='left', padx=3)

        threshold_label = ttk.LabelFrame(methodframe, text="Threshold (0-255)")
        threshold_label.grid(row=1, column=1, sticky='w', padx=10, pady=10)
        self.threshold = ttk.Entry(threshold_label, width=15)
        self.threshold.insert(0, '9')
        self.threshold.pack(side='left', padx=3)

        opening_label = ttk.LabelFrame(methodframe, text="Opening")
        opening_label.grid(row=1, column=2, sticky='w', padx=10, pady=10)
        self.opening = ttk.Entry(opening_label, width=15)
        self.opening.insert(0, '1')
        self.opening.pack(side='left', padx=3)

        closing_label = ttk.LabelFrame(methodframe, text="Closing")
        closing_label.grid(row=1, column=3, sticky='w', padx=10, pady=10)
        self.closing = ttk.Entry(closing_label, width=15)
        self.closing.insert(0, '3')
        self.closing.pack(side='left', padx=3)

        skeletonize_label = ttk.LabelFrame(methodframe, text="Skeletonize")
        skeletonize_label.grid(row=2, column=1, sticky='w', padx=10)
        self.skeletonize = tk.IntVar()
        self.skeletonize_box = tk.Checkbutton(skeletonize_label,
                                    variable=self.skeletonize)
        self.skeletonize_box.pack()

        prune_size_label = ttk.LabelFrame(methodframe, text="Prune size")
        prune_size_label.grid(row=2, column=2, sticky='w', padx=10)
        self.prune_size = ttk.Entry(prune_size_label, width=15)
        self.prune_size.insert(0, '0')
        self.prune_size.pack(side='left', padx=3)

        do_full_prune_label = ttk.LabelFrame(methodframe,
                        text="Full prune")
        do_full_prune_label.grid(row=2, column=3, sticky='w', padx=10)
        self.do_full_prune = tk.IntVar()
        self.do_full_prune_box = tk.Checkbutton(do_full_prune_label,
                                    variable=self.do_full_prune)
        self.do_full_prune_box.pack()


        ###########################

        ######### FILTER  SECTION #########

        filter_frame = ttk.LabelFrame(self, text="Filtering")
        filter_frame.grid(row=2, sticky='w')

        minsize_label = ttk.LabelFrame(filter_frame,
                     text="Minimum size (px)")
        minsize_label.grid(row=0, column=0, sticky='w', padx=10)
        self.minsize = ttk.Entry(minsize_label, width=15)
        self.minsize.insert(0, '25')
        self.minsize.pack(side='left', padx=3)

        maxsize_label = ttk.LabelFrame(filter_frame,
                     text="Maximum size (px)")
        maxsize_label.grid(row=0, column=1, sticky='w', padx=10)
        self.maxsize = ttk.Entry(maxsize_label, width=15)
        self.maxsize.insert(0, '120')
        self.maxsize.pack(side='left', padx=3)

        minimum_ecc_label = ttk.LabelFrame(filter_frame,
                                text="Worm-like (0-1)")
        minimum_ecc_label.grid(row=0, column=2, sticky='w', padx=10)
        self.minimum_ecc = ttk.Entry(minimum_ecc_label, width=15)
        self.minimum_ecc.insert(0, '0.93')
        self.minimum_ecc.pack(side='left', padx=3)

        ###########################


        ######### CUT-OFF SECTION #########

        cut_off_frame = ttk.LabelFrame(self, text="\n Cut-off tools(choose frames in which the number of particles is set as cut-off) and extra filter")
        cut_off_frame.grid(row=3, sticky='w')

        cutoff_filter_label = ttk.LabelFrame(cut_off_frame, text="Cut-off")
        cutoff_filter_label.grid(row=0, column=0, sticky='w', padx=10)
        self.cutoff_filter = tk.IntVar()
        self.cutoff_filter_box = tk.Checkbutton(cutoff_filter_label,
                                    variable=self.cutoff_filter)
        self.cutoff_filter_box.pack()

        use_average_label = ttk.LabelFrame(cut_off_frame, text="Average or Max")
        use_average_label.grid(row=0, column=1, sticky='w', padx=10)
        self.use_average = ttk.Combobox(use_average_label, values=('Average',
                    'Maximum'), state='readonly', width = 8)
        self.use_average.pack(side='left', padx=3)
        self.use_average.current(0)

        lower_label = ttk.LabelFrame(cut_off_frame,
                     text="Start frame")
        lower_label.grid(row=0, column=2, sticky='w', padx=5)
        self.lower = ttk.Entry(lower_label, width=7)
        self.lower.insert(0, '0')
        self.lower.pack(side='left', padx=3)

        upper_label = ttk.LabelFrame(cut_off_frame,
                     text="End frame")
        upper_label.grid(row=0, column=3, sticky='w', padx=5)
        self.upper = ttk.Entry(upper_label, width=7)
        self.upper.insert(0, '100')
        self.upper.pack(side='left', padx=3)

        extra_filter_label = ttk.LabelFrame(cut_off_frame, text="Extra filter")
        extra_filter_label.grid(row=0, column=4, sticky='w', padx=10)
        self.extra_filter = tk.IntVar()
        self.extra_filter_box = tk.Checkbutton(extra_filter_label,
                                    variable=self.extra_filter)
        self.extra_filter_box.pack()

        Bends_max_label = ttk.LabelFrame(cut_off_frame,
                     text="Max Bends")
        Bends_max_label.grid(row=0, column=5, sticky='w', padx=5)
        self.Bends_max = ttk.Entry(Bends_max_label, width=7)
        self.Bends_max.insert(0, '20')
        self.Bends_max.pack(side='left', padx=3)

        Speed_max_label = ttk.LabelFrame(cut_off_frame,
                     text="Max Speed")
        Speed_max_label.grid(row=0, column=6, sticky='w', padx=5)
        self.Speed_max = ttk.Entry(Speed_max_label, width=7)
        self.Speed_max.insert(0, '0.035')
        self.Speed_max.pack(side='left', padx=3)

        ###########################

        ######### TRAJECTORIES  SECTION #########

        trajs_frame = ttk.LabelFrame(self, text="Forming trajectories")
        trajs_frame.grid(row=4, sticky='w', pady=padframes)

        maxdist_label = ttk.LabelFrame(trajs_frame,
                     text="Maximum move distance (px)")
        maxdist_label.grid(row=0, column=0, sticky='w', padx=10)
        self.maxdist = ttk.Entry(maxdist_label, width=15)
        self.maxdist.insert(0, '10')
        self.maxdist.pack(side='left', padx=3)

        minlength_label = ttk.LabelFrame(trajs_frame,
                     text="Minimum length (frames)")
        minlength_label.grid(row=0, column=1, sticky='w', padx=10)
        self.minlength = ttk.Entry(minlength_label, width=15)
        self.minlength.insert(0, '50')
        self.minlength.pack(side='left', padx=3)

        memory_label = ttk.LabelFrame(trajs_frame,
                     text="Memory (frames)")
        memory_label.grid(row=0, column=2, sticky='w', padx=10)
        self.memory = ttk.Entry(memory_label, width=15)
        self.memory.insert(0, '5')
        self.memory.pack(side='left', padx=3)

        ###########################

        ######### BENDS/VELOCITY  SECTION #########

        benvel_frame = ttk.LabelFrame(self, text="Bends and Velocity")
        benvel_frame.grid(row=5, sticky='w')

        bendthres_label = ttk.LabelFrame(benvel_frame,
                     text="Bend threshold")
        bendthres_label.grid(row=0, column=0, sticky='w', padx=10)
        self.bendthres = ttk.Entry(bendthres_label, width=15)
        self.bendthres.insert(0, '2.1')
        self.bendthres.pack(side='left', padx=3)

        minbends_label = ttk.LabelFrame(benvel_frame,
                     text="Minimum bends")
        minbends_label.grid(row=0, column=1, sticky='w', padx=10)
        self.minbends = ttk.Entry(minbends_label, width=15)
        self.minbends.insert(0, '0')
        self.minbends.pack(side='left', padx=3)

        velframes_label = ttk.LabelFrame(benvel_frame,
                     text="Frames to estimate velocity")
        velframes_label.grid(row=0, column=2, sticky='w', padx=10)
        self.velframes = ttk.Entry(velframes_label, width=15)
        self.velframes.insert(0, '49')
        self.velframes.pack(side='left', padx=3)

        ######### TRAJECTORIES  SECTION #########

        dead_frame = ttk.LabelFrame(self, text="Paralyzed worm statistics")
        dead_frame.grid(row=6, sticky='w', pady=10)

        maxbpm_label = ttk.LabelFrame(dead_frame,
                     text="Maximum beat per minute")
        maxbpm_label.grid(row=0, column=0, sticky='w', padx=10)
        self.maxbpm = ttk.Entry(maxbpm_label, width=15)
        self.maxbpm.insert(0, '0.5')
        self.maxbpm.pack(side='left', padx=3)

        maxvel_label = ttk.LabelFrame(dead_frame,
                     text="Maximum velocity (mm/s)")
        maxvel_label.grid(row=0, column=1, sticky='w', padx=10)
        self.maxvel = ttk.Entry(maxvel_label, width=15)
        self.maxvel.insert(0, '0.1')
        self.maxvel.pack(side='left', padx=3)

        ###########################

        ######## REGION OF INTERESTS SECTION #########
        self.adding_roi = False

        roi_frame = ttk.LabelFrame(self, text="Region of interests")
        roi_frame.grid(row=7, sticky='w')

        roi_btn_add = ttk.Button(roi_frame, text="Add new",
                            command=self.add_roi)
        roi_btn_add.pack(side='left', padx=15)

        self.rois = ttk.Combobox(roi_frame, state='disabled')
        self.rois.pack(side='left', padx=3)
        self.rois['values'] = []

        self.roi_btn_show = ttk.Button(roi_frame, text="Show",
                            command=self.show_roi, state='disabled')
        self.roi_btn_show.pack(side='left', padx=5)

        self.roi_btn_edit = ttk.Button(roi_frame, text="Redraw",
                            command=self.edit_roi, state='disabled')
        self.roi_btn_edit.pack(side='left', padx=5)

        self.roi_btn_del = ttk.Button(roi_frame, text="Delete",
                            command=self.del_roi, state='disabled')
        self.roi_btn_del.pack(side='left', padx=5)

        ###########################

        ######## OUTPUT SECTION #########

        output_frame = ttk.LabelFrame(self, text="Output")
        output_frame.grid(row=8, sticky='w')

        outputdirframe = ttk.Frame(output_frame)
        outputdirframe.grid(row=0)
        self.outputname = ttk.Entry(outputdirframe, width=80, state='readonly')

        self.outputname.grid(row=0, padx=10)
        output_btn = ttk.Button(outputdirframe, text="Browse",
                    command=self.find_outputfolder)
        output_btn.grid(row=0, column=1, padx=5)

        outputinfo_frame = ttk.Frame(output_frame)
        outputinfo_frame.grid(row=1, sticky='w')

        outputframes_label = ttk.LabelFrame(outputinfo_frame,
                text="Output frames")
        outputframes_label.grid(row=2, sticky='w', padx=10)
        self.outputframes = ttk.Entry(outputframes_label, width=15)
        self.outputframes.insert(0, '0')
        self.outputframes.pack(side='left', padx=3)

        font_size = ttk.LabelFrame(outputinfo_frame, text="Font size")
        font_size.grid(row=2, column=1, sticky='w', padx=10)
        self.font_size = ttk.Entry(font_size, width=15)
        self.font_size.insert(0, '8')
        self.font_size.pack(side='left', padx=3)

        ######### Buttons #########
        add_job_btn_frame = ttk.Frame(self)
        add_job_btn_frame.grid(row=9, sticky='W')
        if editing:
            add_job_btn = ttk.Button(add_job_btn_frame, text="Confirm edits",
                                command=self.add_job)
        else:
            add_job_btn = ttk.Button(add_job_btn_frame, text="Add job",
                                command=self.add_job)
        add_job_btn.pack(side='left', padx=5)
        cancel_btn = ttk.Button(add_job_btn_frame, text="Cancel",
                            command=self.destroy)
        cancel_btn.pack(side='left', padx=5)

        ###########################

        ########## EDITING #################
        if editing:
            self.update_video_info(edit_job['video'])

            # Entries
            self.start_frame.delete(0,'end')
            self.start_frame.insert(0,edit_job['startframe'])
            self.use_frame.delete(0,'end')
            self.use_frame.insert(0,edit_job['useframes'])
            self.fps.delete(0,'end')
            self.fps.insert(0,edit_job['fps'])
            self.px_to_mm.delete(0,'end')
            self.px_to_mm.insert(0,edit_job['px_to_mm'])
            self.z_use.delete(0,'end')
            self.z_use.insert(0,edit_job['z_use'])
            self.z_padding.delete(0,'end')
            self.z_padding.insert(0,edit_job['z_padding'])
            self.std_px.delete(0,'end')
            self.std_px.insert(0,edit_job['std_px'])
            self.threshold.delete(0,'end')
            self.threshold.insert(0,edit_job['threshold'])
            self.opening.delete(0,'end')
            self.opening.insert(0,edit_job['opening'])
            self.closing.delete(0,'end')
            self.closing.insert(0,edit_job['closing'])
            self.minsize.delete(0,'end')
            self.minsize.insert(0,edit_job['minsize'])
            self.maxsize.delete(0,'end')
            self.maxsize.insert(0,edit_job['maxsize'])
            self.maxdist.delete(0,'end')
            self.maxdist.insert(0,edit_job['maxdist'])
            self.minlength.delete(0,'end')
            self.minlength.insert(0,edit_job['minlength'])
            self.memory.delete(0,'end')
            self.memory.insert(0,edit_job['memory'])
            self.bendthres.delete(0,'end')
            self.bendthres.insert(0,edit_job['bendthres'])
            self.minbends.delete(0,'end')
            self.minbends.insert(0,edit_job['minbends'])
            self.velframes.delete(0,'end')
            self.velframes.insert(0,edit_job['velframes'])
            self.maxbpm.delete(0,'end')
            self.maxbpm.insert(0,edit_job['maxbpm'])
            self.maxvel.delete(0,'end')
            self.maxvel.insert(0,edit_job['maxvel'])
            self.outputframes.delete(0,'end')
            self.outputframes.insert(0,edit_job['outputframes'])
            self.font_size.delete(0,'end')
            self.font_size.insert(0,edit_job['font_size'])
            self.minimum_ecc.delete(0,'end')
            self.minimum_ecc.insert(0,edit_job['minimum_ecc'])
            self.prune_size.delete(0,'end')
            self.prune_size.insert(0,edit_job['prune_size'])
            self.lower.delete(0, 'end')#new
            self.lower.insert(0, edit_job['lower'])#new
            self.upper.delete(0, 'end') #new
            self.upper.insert(0, edit_job['upper'])#new
            self.Bends_max.delete(0, 'end')#new
            self.Bends_max.insert(0, edit_job['Bends_max'])#new
            self.Speed_max.delete(0, 'end')#new
            self.Speed_max.insert(0, edit_job['Speed_max'])#new
        

            # Rest:
            self.darkfield.set(edit_job['darkfield'])
            self.skeletonize.set(edit_job['skeletonize'])
            self.do_full_prune.set(edit_job['do_full_prune'])
            self.extra_filter.set(edit_job['extra_filter']) # new
            self.cutoff_filter.set(edit_job['cutoff_filter']) #new
            self.use_average.current(1 if edit_job['use_average'] == 'Maximum' else 0) # new
            self.method.current(1 if edit_job['method']=='Keep Dead' else 0)
            self.outputname.configure(state='normal')
            self.outputname.insert(0,edit_job['outputname'])
            self.outputname.configure(state='readonly')

            # Region of interests:
            self.rois.configure(state='normal')
            self.regions = edit_job['regions']
            self.rois['values'] = edit_job['regions'].keys()
            if len(edit_job['regions'])>0:
                self.rois.set(edit_job['regions'].keys()[0])
                self.roi_btn_show.configure(state='normal')
                self.roi_btn_edit.configure(state='normal')
                self.roi_btn_del.configure(state='normal')
            self.rois.configure(state='readonly')

    def add_to_job(self, job, fieldname, inputfield, typeconv):
        string = inputfield.get()
        if isinstance(string, int):
            typed = typeconv(string)
        else:
            if len(string)==0:
                err = "Field '" + fieldname + "' empty!"
                tkMessageBox.showerror('Error', err)
                return False
            try:
                typed = typeconv(string)
            except:
                err = "Error in field '" + fieldname + "'"
                tkMessageBox.showerror('Error', err)
                return False
        job[fieldname] = typed
        return True

    def add_job(self):
        job = {}
        add = self.add_to_job
        if not add(job, 'video', self.videoname, str): return
        if not add(job, 'startframe', self.start_frame, int): return
        if not add(job, 'useframes', self.use_frame, int): return
        if not add(job, 'fps', self.fps, float): return
        if not add(job, 'px_to_mm', self.px_to_mm, float): return
        if not add(job, 'darkfield', self.darkfield, bool): return
        if not add(job, 'extra_filter', self.extra_filter, bool): return #new
        if not add(job, 'cutoff_filter', self.cutoff_filter, bool): return #new
        if not add(job, 'method', self.method, str): return
        if not add(job, 'use_average', self.use_average, str): return #new
        if not add(job, 'z_use', self.z_use, int): return
        if not add(job, 'z_padding', self.z_padding, int): return
        if not add(job, 'std_px', self.std_px, int): return
        if not add(job, 'threshold', self.threshold, int): return
        if not add(job, 'opening', self.opening, int): return
        if not add(job, 'closing', self.closing, int): return
        if not add(job, 'skeletonize', self.skeletonize, bool): return
        if not add(job, 'do_full_prune', self.do_full_prune, bool): return
        if not add(job, 'prune_size', self.prune_size, int): return
        if not add(job, 'lower', self.lower, int): return #new
        if not add(job, 'upper', self.upper, int): return #new
        if not add(job, 'Bends_max', self.Bends_max, float): return #new
        if not add(job, 'Speed_max', self.Speed_max, float): return #new
        if not add(job, 'minsize', self.minsize, int): return
        if not add(job, 'maxsize', self.maxsize, int): return
        if not add(job, 'maxdist', self.maxdist, int): return
        if not add(job, 'minlength', self.minlength, int): return
        if not add(job, 'memory', self.memory, int): return
        if not add(job, 'bendthres', self.bendthres, float): return
        if not add(job, 'minbends', self.minbends, float): return
        if not add(job, 'velframes', self.velframes, int): return
        if not add(job, 'maxbpm', self.maxbpm, float): return
        if not add(job, 'maxvel', self.maxvel, float): return
        if not add(job, 'outputname', self.outputname, str): return
        if not add(job, 'outputframes', self.outputframes, int): return
        if not add(job, 'minimum_ecc', self.minimum_ecc, float): return
        if not add(job, 'font_size', self.font_size, int): return

        job['regions'] = self.regions
        if self.editing:
            name = self.videoname.get().split('/')[-1]
            self.parent.job_buttons[self.edit_index]['thisframe'].grid_forget()
            self.parent.log('Edited job "%s".'%name)
            del self.parent.job_buttons[self.edit_index]
            del self.parent.jobs[self.edit_index]
        self.parent.add_job(job)
        self.destroy()

    def update_video_info(self, filename,filenames=None):
        if filenames == None:
            filenames = (filename,)
        try:
            video = cv2.VideoCapture(filename)
            n_frames = video.get(cv.CV_CAP_PROP_FRAME_COUNT)
        except:
            self.parent.log('Error opening video: '+filename)
            return
        if n_frames < 0.5:
            self.parent.log('Error opening video: '+filename)
            return
        width = video.get(cv.CV_CAP_PROP_FRAME_WIDTH)
        height = video.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
        fps = video.get(cv.CV_CAP_PROP_FPS)

        self.videoname.config(state='normal')
        self.videoname.delete(0, tk.END)
        self.videoname.insert(0, ", ".join(filenames))
        self.videoname.config(state='readonly')
        self.videoinfo.config(text='Size: %dx%d    Number of frames: %d'
           '    Frames per second guesstimate: %d' \
                %(width, height, n_frames, fps))
        self.use_frame.delete(0, tk.END)
        self.use_frame.insert(0, '%d'%n_frames)
        self.fps.delete(0, tk.END)
        self.fps.insert(0, '%d'%fps)

    def find_video(self):
        filenames = tkFileDialog.askopenfilenames()
        if len(filenames)==0:
            return
        if "," in "".join(filenames):
            self.parent.log("Video paths cannot contain commas.")
            return
        filename = filenames[0]
        if filename:
            self.update_video_info(filename, filenames)
        self.grab_set()

    def add_roi(self):
        self.adding_roi = True
        roi = Roi(self)

    def show_roi(self):
        name = self.rois.get()
        region = self.regions[name]
        try:
            video = cv2.VideoCapture(self.videoname.get().split(", ")[0])
            ret, frame = video.read()
        except:
            ret = False
        if not ret:
            self.log('Select a movie first.')
            return
        try:
            start = int(self.start_frame.get())
            end = int(self.use_frame.get())
        except:
            start = 0
            end = video.get(cv.CV_CAP_PROP_FRAME_COUNT)-1
        mid = int((end-start)//2)
        video.set(cv.CV_CAP_PROP_POS_FRAMES, mid)
        ret, frame = video.read()

        plt.figure(figsize=(12, 9.5))
        plt.imshow(frame)
        plt.plot(region['x'], region['y'],'b')
        plt.plot([region['x'][0],region['x'][-1]],
                 [region['y'][0],region['y'][-1]],'b')
        axes = (0,frame.shape[1],frame.shape[0],0)
        plt.axis(axes)
        plt.show()

    def del_roi(self):
        if tkMessageBox.askquestion("Delete", "Are You Sure?", icon='warning'):
            name = self.rois.get()
            self.rois['values'] = [k for k in self.rois['values'] if k!=name]
            del self.regions[name]
            if len(self.rois['values'])>0:
                self.rois.set(self.rois['values'][0])
            else:
                self.rois.set('')
                self.roi_btn_show.configure(state='disabled')
                self.roi_btn_edit.configure(state='disabled')
                self.roi_btn_del.configure(state='disabled')

    def edit_roi(self):
        self.adding_roi = False
        roi = Roi(self)

    def find_outputfolder(self):
        filename = tkFileDialog.asksaveasfilename(filetypes=
                    [('','Directory name')])
        if filename:
            self.outputname.config(state='normal')
            self.outputname.delete(0, tk.END)
            self.outputname.insert(0, filename)
            self.outputname.config(state='readonly')

class Roi(tk.Toplevel):
    def closing(self):
        if len(self.xx)<=2:
            self.destroy()
            self.parent.grab_set()
            return

        self.parent.rois.configure(state='normal')
        self.parent.roi_btn_show.configure(state='normal')
        self.parent.roi_btn_edit.configure(state='normal')
        self.parent.roi_btn_del.configure(state='normal')

        if self.parent.adding_roi:
            name = None
            while not isinstance(name, str):
                name = tkSimpleDialog.askstring('Region name',
                        'Input name of region')
            self.parent.rois['values'] = list(self.parent.rois['values']) \
                                                +[name]
            self.parent.rois.set(name)
        else:
            name = self.parent.rois.get()

        self.parent.regions[name] = {'x' : self.xx, 'y' : self.yy}

        self.parent.rois.configure(state='readonly')
        self.destroy()
        self.parent.grab_set()

    def __init__(self, parent,  *args, **kwargs):
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.protocol('WM_DELETE_WINDOW', self.closing)
        self.wm_title("Region of Interest")
        self.grab_set()
        self.parent = parent

        self.f = matplotlib.figure.Figure(figsize=(12, 9.5))
        self.ax = self.f.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        try:
            video = cv2.VideoCapture(parent.videoname.get().split(", ")[0])
            ret, frame = video.read()
        except:
            ret = False
        if not ret:
            parent.parent.log('Select a movie first.')
            self.destroy()
            return
        try:
            start = int(parent.start_frame.get())
            end = int(parent.use_frame.get())
        except:
            start = 0
            end = video.get(cv.CV_CAP_PROP_FRAME_COUNT)-1
        mid = int((end-start)//2)
        video.set(cv.CV_CAP_PROP_POS_FRAMES, mid)
        ret, frame = video.read()

        self.ax.imshow(frame)
        axes = (0,frame.shape[1],frame.shape[0],0)
        self.ax.axis(axes)
        self.xx = []
        self.yy = []
        poly = [1]
        def onclick(event):
            #add = (event.button==1)
            x, y = event.xdata, event.ydata
            if x != None:
                self.xx.append(x)
                self.yy.append(y)
                self.ax.plot(self.xx, self.yy, '-xb')
                if len(self.xx)>=3:
                    if poly[0]!=1:
                        poly[0].pop(0).remove()
                    poly[0] = self.ax.plot([self.xx[0],self.xx[-1]],[self.yy[0],self.yy[-1]],'--b')
                self.ax.axis(axes)
                self.canvas.draw()
        cid = self.canvas.mpl_connect('button_press_event', onclick)

if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
