
from multiprocessing import Pool
import time
 
def unwrap_self_f(arg, **kwarg):
    return C.f(*arg, **kwarg)
 
class C:
    def f(self, name):
        print ('hello %s,'%name)
        time.sleep(5)
        print('nice to meet you.')
     
    def run(self):
        pool = Pool(processes=2)
        names = ('frank', 'justin', 'osi', 'thomas')
        pool.apply_async(unwrap_self_f, zip([self]*len(names), names))
        pool.close()
        pool.join()
 
if __name__ == '__main__':
    c = C()
    c.run()