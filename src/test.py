"""
cpu_gpu.py
An OpenCL-OpenCV-Python CPU vs GPU comparison
"""
import cv2
import timeit

# A simple image pipeline that runs on both Mat and Umat
def img_cal(img, mode):
    if mode=='UMat':
        img = cv2.UMat(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 1.5)
    img = cv2.Canny(img, 0, 50)
    if type(img) == 'cv2.UMat': 
        img = cv2.UMat.get(img)
    return img

# Timing function
def run(processor, function, n_threads, N):
    cv2.setNumThreads(n_threads)
    t = timeit.timeit(function, globals=globals(), number=N)/N*1000
    print('%s avg. with %d threads: %0.2f ms' % (processor, n, t))
    return t

img = cv2.imread('ct.tif') 
N = 1000
n_threads = [1,  16]

processor = {'GPU': "img_cal(img_UMat)", 
             'CPU': "img_cal(img)"}
results = {}
for n in n_threads: 
    for pro in processor.keys():
        results[pro,n] = run(processor=pro, 
                             function= processor[pro], 
                             n_threads=n, N=N)

print('\nGPU speed increase over 1 CPU thread [%%]: %0.2f' % \
      (results[('CPU', 1)]/results[('GPU', 1)]*100))
print('CPU speed increase on 4 threads versus 1 thread [%%]: %0.2f' % \
      (results[('CPU', 1)]/results[('CPU', 16)]*100))
print('GPU speed increase versus 4 threads [%%]: %0.2f' % \
      (results[('CPU', 4)]/results[('CPU', 1)]*100))