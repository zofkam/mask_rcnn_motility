### These three lists must be the same length
filenames = ['sql_0uM_controls.avi']
save_as = ['sql_0uM_controls']
fps = [20]

###### Do not change below this line ######
import os
from multiprocessing import Pool, cpu_count

def runf(s):
	os.system('python multiwormtracker.py %s/settings.py'%s)

if __name__ == '__main__':
	pool = Pool(cpu_count())
	f = open('settings.py')
	settings = f.read().split('\n')
	f.close()
	for i in xrange(len(filenames)):
		save_as[i] = save_as[i].rstrip('/')+'/'
		try:
			os.mkdir(save_as[i])
		except OSError:
			pass

		for j,s in enumerate(settings):
			if s.startswith('filename'):
				settings[j] = "filename = '%s'"%filenames[i]
			if s.startswith('save_as'):
				settings[j] = "save_as = '%s'"%save_as[i]
			if s.startswith('fps'):
				settings[j] = "fps = "+str(fps[i])
		f = open('%ssettings.py'%save_as[i],'w')
		f.write("\n".join(settings))
		f.close()
		print ' ******** Loaded %s *******'%save_as[i]
		print '          File: %s'%filenames[i]
		print '          FPS = '+str(fps[i])
		print

	pool.map(runf,save_as,chunksize=1)

