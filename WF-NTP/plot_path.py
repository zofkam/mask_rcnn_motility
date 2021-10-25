import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle
import tkFileDialog

class Video:
    def __init__(self,fname,grey=False):
        self.cap = cv2.VideoCapture(fname)
        self.fname = fname
        self.name = "".join(fname.split(".")[:-1]).replace('/','_')
        self.len = self.cap.get(cv.CV_CAP_PROP_FRAME_COUNT) - start_frame
        if limit_images_to and limit_images_to<(self.len-start_frame):
            self.len = limit_images_to
        self.grey = grey
        if grey:
            for _ in xrange(100):
                ret, frame = self.cap.read()
                if ret:
                    break
            if len(frame.shape)==2:
                self.grey = False
            self.cap.set(cv.CV_CAP_PROP_POS_FRAMES,0)
    def next(self):
        ret = False
        for _ in xrange(100):
            ret, frame = self.cap.read()
            if ret:
                break
            time.sleep(0.1*random.random())
        if ret:
            if self.grey:
                return frame[:,:,0]
            else:
                return frame
        else:
            raise StopIteration
    def set_index(self,i):
        self.cap.set(cv.CV_CAP_PROP_POS_FRAMES,i)
    def restart(self):
        self.set_index(start_frame)
    def __getitem__(self,i):
        if i<0:
            i += self.len
        self.set_index(start_frame+i)
        return self.next()
    def __len__(self):
        return int(self.len)
    def release(self):
        self.cap.release()


def plot_path(filename):
    saved_name = "/".join(filename.split('/')[:-1])

    plt.close()
    plt.figure(figsize=(10,8))
    colormap = cm.Set2

    with open(filename) as f:
        track = cPickle.load(f)

    particles = set(track['particle'])
    colours = [colormap(i/float(len(particles)))
                    for i in xrange(len(particles))]
    rand = np.random.permutation(len(particles))
    for i, p in enumerate(particles):
        idx = track['particle'] == p

        x = track['x'][idx]
        y = track['y'][idx]

        plt.plot(y,x,c=colours[rand[i]], linewidth=2.5)
    plt.axis('equal')
    a = list(plt.axis())
    a[3],a[2] = a[2], a[3]
    plt.axis(a)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #plt.axis('off')

    plt.savefig(saved_name+'/track.png')
    plt.savefig(saved_name+'/track.pdf')

    plt.show()

if __name__ == '__main__':
    filename = tkFileDialog.askopenfilename(title='Locate a track.p file',filetypes=[("track.p file","*.p")])
    plot_path(filename)
