#pragma once
#include <unordered_map>
#include <string>
#include <ctime>
#include "cuda_settings.h"
using namespace std;

class CPU_Timer{
        private:
          unordered_map<string, float> cmap;
          unordered_map<string, int> counter;
          unordered_map<string, clock_t> start;
        public:
        CPU_Timer(){}
        
        void start_clock(string name){
            if(!cmap.count(name)){
                counter[name]=0;
                cmap[name]=0.0;
            }
            start[name] = clock();
        }
        void stop_clock(string name){
            if(cmap.count(name)){
                cmap[name]+=clock()-start[name];
                counter[name]+=1;
            } else{
                // printf("Invalid clock!");
                cout << "Invalid clock name! " << name << endl;
                // exit(-1);
            }
        }

        void print_clock(string name){
            if(cmap.count(name)){
                printf("%30s : %15.2fs CPU\t(%9d calls)\n",name.c_str(),cmap[name]/CLOCKS_PER_SEC,counter[name]);
            }
        }
    };

class GPU_Timer{
        private:
          unordered_map<string, float> cmap;
          unordered_map<string, int> counter;
          unordered_map<string, cudaEvent_t> start;
          unordered_map<string, cudaEvent_t> stop;
        public:
        GPU_Timer(){}
        
        void start_clock(string name){
            if(!cmap.count(name)){
                cudaEvent_t start_;
                cudaEvent_t stop_;
                counter[name]=0;
                CUDA_CHECK(cudaEventCreate(&start_));
                CUDA_CHECK(cudaEventCreate(&stop_));
                start[name]=start_;
                stop[name]=stop_;
                cmap[name]=0.0;
            }
            CUDA_CHECK(cudaEventRecord(start[name]));
        }
        void stop_clock(string name){
            if(cmap.count(name)){
                CUDA_CHECK(cudaEventRecord(stop[name]));
                CUDA_CHECK(cudaEventSynchronize(stop[name]));
                float elapsed_time;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start[name], stop[name]));
                cmap[name]+=elapsed_time;
                counter[name]+=1;
            } else{
                // printf("Invalid clock name!, %",name);
                cout << "Invalid clock name! " << name << endl;
                // exit(-1);
            }
        }

        void print_clock(string name){
            if(cmap.count(name)){
                printf("%30s : %15.2fs GPU\t(%9d calls)\n",name.c_str(),cmap[name]/1000.0,counter[name]);
            }
        }
};