
import concurrent.futures
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import read_data as read
import random_init as rin
import compute_dis_mat as cdm
import compute_pathlen as cpl
import compute_paths as cp
import eval_particals as evalp
import threading
import time
import multiprocessing

class PSO(object):
    def __init__(self, num_city, data):
        self.iter_max = 5000  # iteration number
        self.num = 50  # partical number
        self.num_city = num_city  # city number
        self.location = data # location of city 
        self.dis_mat = cdm.compute_dis_mat(self.num_city, self.location)  # compute the distance between cities
        # initialize all the particals
        self.particals = rin.random_init(self.num, self.num_city)
        self.lenths = cp.compute_paths(self.particals,self.dis_mat)
        # generate the initial solution
        init_l = min(self.lenths)
        #global glo_best_len 
        #glo_best_len = init_l
        init_index = self.lenths.index(init_l)
        self.init_path = self.particals[init_index]
        #global glo_best_path 
        #glo_best_path = self.init_path
        # draw the initial path
        init_show = self.location[self.init_path]
        plt.subplot(1, 2, 1)
        plt.title('init best result')
        plt.plot(init_show[:, 0], init_show[:, 1])
        # record the local best solution
        self.local_best = self.particals
        self.local_best_len = self.lenths
        # record the global best solution
        self.global_best = self.init_path
        self.global_best_len = init_l
        # output the best solution
        self.best_l = self.global_best_len
        self.best_path = self.global_best
        #self.iter_x = [0]
        #self.iter_y = [init_l]
        #define a lock to ensure that only one thread is performing variable access  
        self.mutex = threading.Lock()
        self.count1=0
        self.count2=0

    

     # cross particals
    def cross(self, cur, best):
        one = cur.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l,2)
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for t in one:
            if t in cross_part:
                continue
            tmp.append(t)
        # two methods
        one = tmp + cross_part
        l1 = cpl.compute_pathlen(one, self.dis_mat)
        one2 = cross_part + tmp
        l2 = cpl.compute_pathlen(one2, self.dis_mat)
        if l1<l2:
            return one, l1
        else:
            return one, l2


    # particals mutation
    def mutate(self, one):
        one = one.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)
        one[x], one[y] = one[y], one[x]
        l2 = cpl.compute_pathlen(one,self.dis_mat)
        return one, l2

    

    # main pso
    def psorun1(self):

        
        for cnt1 in range(0 , 1):
            # Update particle swarm
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]
                # Cross with the current local optimal solution
                new_one, new_l = self.cross(one, self.local_best[i])
                if new_l < self.best_l:    
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l

            # Cross with the current global optimal solution
                global glo_best_path
                new_one, new_l = self.cross(one, glo_best_path)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one  
                    tmp_l = new_l
                # mutation
                one, tmp_l = self.mutate(one)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l

                # update partical
                self.particals[i] = one
                self.lenths[i] = tmp_l
            # Evaluate the particle swarm, update the individual's local optimal and the individual's global optimal position
            evalp.eval_particals(self)
            # update the output solution
            #self.mutex.acquire()
            if self.global_best_len < self.best_l:
                #self.best_l = self.global_best_len
                #self.best_path = self.global_best
                global glo_best_len
                glo_best_len = self.global_best_len
                self.best_l = self.global_best_len
                glo_best_path = self.global_best
            #self.mutex.release()
            #print(cnt)
        
            #self.iter_x.append(cnt)
            #self.iter_y.append(self.best_l)
            
            self.count1 = self.count1 + 1 
            print(self.count1)



if __name__ == "__main__":
    # read city location data
    data = read.read_tsp('C:/Users/gh/Desktop/psotsp/data/70.tsp')


    data = np.array(data)
    plt.suptitle('PSO in 70.tsp')
    data = data[:, 1:]
    psor = PSO(num_city=data.shape[0], data=data.copy())
    
    
    glo_best_len = 0
    dis_mat = cdm.compute_dis_mat(data.shape[0], data.copy())
    particals = rin.random_init(150, data.shape[0])
    lenths = cp.compute_paths(particals,dis_mat)
    init_l = min(lenths)
    init_index = lenths.index(init_l)
    init_path = particals[init_index]
    glo_best_path =  init_path
    
    
    
    
# with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
#     futures = [executor.submit(psor.psorun1)]





    pool = multiprocessing.Pool(processes = 8)
    time_start=time.time()
    for i in range(100):
        pool.apply_async(psor.psorun1())

    pool.close()
    pool.join()

    time_end=time.time()
    print('totally cost',time_end-time_start)
    #Best_path, Best = pso.run()
    
    print(glo_best_len)
    plt.subplot(1, 2, 2)
    
    Best_path = np.vstack([psor.location[glo_best_path], psor.location[glo_best_path][0]])
    plt.plot(Best_path[:, 0], Best_path[:, 1])
    plt.title('result')
    plt.show()
