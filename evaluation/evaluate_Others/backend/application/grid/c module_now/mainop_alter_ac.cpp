//example.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <utility>
#include <ctime>

//#include <windows.h>
//#undef max
//#undef min

#include "utils/base.h"
#include "utils/simpleCluster.h"
#include "utils/lap.h"
#include "utils/util.h"
#include "utils/optimize.h"
#include "utils/treemap.h"
#include "convexity/measureEdge.h"
#include "convexity/measureTriples.h"
#include "convexity/measureCHS.h"
#include "convexity/measureCHC.h"
#include "convexity/newMeasure2020.h"
#include "convexity/measureDoubles.h"
#include "convexity/newMeasureTB.h"
#include "convexity/newMeasureBoundary.h"
#include "convexity/newMeasureDeviation.h"
#include "convexity/newMeasureAlphaTriples.h"
#include "convexity/newMeasureMoS.h"
#include "convexity/measureTriplesFree.h"
#include "convexity/measureCHCFree.h"

// simple cluster, python interface
std::vector<int> Optimizer::getClusters(
const std::vector<int> &_grid_asses,
const std::vector<int> &_labels) {
    int N = _grid_asses.size();
    int num = _labels.size();
    int square_len = ceil(sqrt(N));
    int maxLabel = 0;
    for(int i=0;i<num;i++)maxLabel = std::max(maxLabel, _labels[i]+1);

    int *grid_asses = new int[N];
    int *labels = new int[num];
    int *cluster_labels = new int[num];
    for(int i=0;i<N;i++)grid_asses[i] = _grid_asses[i];
    for(int i=0;i<num;i++)labels[i] = _labels[i];

    getClustersArray(cluster_labels, maxLabel+5, int(0.02*N), grid_asses, labels, N, num, square_len);

    std::vector<int> ret(num, 0);
    for(int i=0;i<num;i++)ret[i] = cluster_labels[i];

    delete[] grid_asses;
    delete[] labels;
    delete[] cluster_labels;
    return ret;
}

// lap jv solve bi-graph-match
std::vector<int> Optimizer::solveLapArray(
    const double dist[],
    const int N,
    int k=50) {
    float *cost_matrix = new float[N*N];
    #pragma omp parallel for num_threads(THREADS_NUM)
    for(int i1=0;i1<N;i1++){
        int bias = i1*N;
        for(int i2=0;i2<N;i2++){
            int i = bias+i2;
            cost_matrix[i] = dist[i]+0.5;   // prevent 0 distance
        }
    }
    float *u = new float[N];
    float *v = new float[N];
    int *grid_asses = new int[N];
    int *element_asses = new int[N];
    float cost = lap(N, cost_matrix, false, grid_asses, element_asses, u, v, std::min(k,N));

    std::vector<int> ret(N, 0);
    for(int i=0;i<N;i++)ret[i] = grid_asses[i];

    delete[] cost_matrix;
    delete[] u;
    delete[] v;
    delete[] grid_asses;
    delete[] element_asses;

    return ret;
}

//lap jv, python interface
std::vector<int> Optimizer::solveLap(
    const std::vector<std::vector<double>> &_dist,
    const bool use_knn=false, int k_value=50) {

    int n_row = _dist.size();
    int n_col = _dist[0].size();

    double *dist = new double[n_row*n_col];
    for(int i=0;i<n_row;i++)
    for(int j=0;j<n_col;j++)dist[i*n_col+j]=_dist[i][j];
    if(!use_knn)k_value = n_col;
    std::vector<int> ret = solveLapArray(dist, n_row, k_value);

    delete[] dist;
    return ret;
}

// KM, deal the augmenting path
void Optimizer::workback(int x,int lb[][3], int p2[])
{
    while(x>1){ int y=lb[x][0]; x=lb[x][2]; p2[y]=lb[x][1]; }
}

// KM
std::vector<int> Optimizer::solveKMArray(
    const double dist[],
    const int n_row, const int n_col) {
    const double tiny = 0.000001;

    double amax=0;
    for(int i=0;i<n_row;i++){
        int bias = i*n_col;
        for(int j=0;j<n_col;j++){
            amax=std::max(dist[bias+j],amax);
        }
    }

    double *a = new double[n_row*n_col];    // dist matrix -> award matrix
    for(int i=0;i<n_row;i++) {
        int bias = i*n_col;
        for(int j=0;j<n_col;j++) {
            a[bias+j] = amax-dist[bias+j];
        }
    }

    double* d1=new double[n_row+5];    // vertex labels, left part
    double* d2=new double[n_col+5];    // vertex labels, right part
    for(int i=0;i<n_row;i++)
    {
        int bias = i*n_col;
        d1[i]=0;
        for(int j=0;j<n_col;j++)
            d1[i]=std::max(d1[i],a[bias+j]);
    }
    for(int i=0;i<n_col;i++)
        d2[i]=0;

    int* f1=new int[n_col+5];    // minarg x in d1[x]+d2[i]-a[x*n_col+i]
    int* f2=new int[n_col+5];    // position of x in queue
    for(int i=0;i<n_col+5;i++)f1[i]=-1;
    for(int i=0;i<n_col+5;i++)f2[i]=-1;

    int N=std::max(n_row,n_col)+5;
    int (*lb)[3]=new int[N+5][3];    // vertex queue in km algorithm
    int l,r;

    int* bo1=new int[n_row+5];    // if been accessed, left part
    int* bo2=new int[n_col+5];    // if been accessed, right part
    for(int i=0;i<n_row+5;i++)bo1[i]=-1;
    for(int i=0;i<n_col+5;i++)bo2[i]=-1;

    int* p1=new int[n_row+5];    // match object, left part
    int* p2=new int[n_col+5];    // match object, right part
    for(int i=0;i<n_row+5;i++)p1[i]=-1;
    for(int i=0;i<n_col+5;i++)p2[i]=-1;

    for(int ii=0;ii<n_row;ii++)    // km algorithm, find match object for left part vertex
    {
        for(int i=0;i<n_col;i++)f1[i]=f2[i]=-1;
        l=0; r=1; lb[r][1]=ii; lb[r][2]=0;
        while(r>0)    // loop until find the match object of ii
        {
            int flag=-1;
            while(l<r)    // queue iterate
            {
                int x;
                l++; x=lb[l][1]; bo1[x]=ii;
                for(int i=0;i<n_col;i++)if(bo2[i]<ii)    // search the right part
                {
                    if(abs(d1[x]+d2[i]-a[x*n_col+i])<tiny)
                    {
                        bo2[i]=ii;
                        if(p2[i]==-1){ p2[i]=x; flag=l; break; }
                        r++; lb[r][0]=i; lb[r][1]=p2[i]; lb[r][2]=l;
                    }else
                    if((f1[i]==-1)||(d1[x]+d2[i]-a[x*n_col+i]<d1[f1[i]]+d2[i]-a[f1[i]*n_col+i])){ f1[i]=x; f2[i]=l; }
                }
                if(flag!=-1)break;
            }
            if(flag!=-1){ workback(flag,lb,p2); break; }


            double dd=2147483647;
            int dd1, dd2;
            for(int i=0;i<n_col;i++)if((bo2[i]<ii)&&(f1[i]>=0))    // find minist decrease
            if(d1[f1[i]]+d2[i]-a[f1[i]*n_col+i]<dd){
                dd = d1[f1[i]]+d2[i]-a[f1[i]*n_col+i];
                dd1 = f1[i];
                dd2 = i;
            }
            for(int i=0;i<n_row;i++)if(bo1[i]==ii)d1[i]-=dd;    // update vertex label, left part
            for(int i=0;i<n_col;i++)if(bo2[i]==ii)d2[i]+=dd;    // update vertex label, right part
            for(int i=0;i<n_col;i++) {
                if((bo2[i]<ii)&&(f1[i]>=0)&&(abs(d1[f1[i]]+d2[i]-a[f1[i]*n_col+i])<tiny))
                {
                    bo2[i]=ii;
                    if(p2[i]==-1){
                        p2[i]=f1[i]; flag=f2[i]; break;
                    }
                    r++; lb[r][0]=i; lb[r][1]=p2[i]; lb[r][2]=f2[i];
                }
            }
            if(flag!=-1){ workback(flag,lb,p2); break; }    // deal the augmenting path
        }
    }

    double ans=0,ans1=0,ans2=0;
    for(int i=0;i<n_row;i++)ans+=d1[i];
    for(int i=0;i<n_col;i++)ans+=d2[i];
//    for(int i=0;i<n_col;i++)ans1+=a[p2[i]*n_col+i];
    for(int i=0;i<n_col;i++)if(p2[i]>=0)p1[p2[i]]=i;
    for(int i=0;i<n_row;i++)ans2+=dist[i*n_col+p1[i]];

    std::vector<int> ret(n_row, 0);
    for(int i=0;i<n_row;i++)ret[i] = p1[i];

    delete[] a;
    delete[] d1;
    delete[] d2;
    delete[] f1;
    delete[] f2;
    delete[] bo1;
    delete[] bo2;
    delete[] p1;
    delete[] p2;
    delete[] lb;

    return ret;
}


// KM
std::vector<int> Optimizer::solveKMArrayLabel(
    const double dist[],
    const int n_row, const int n_col,
    const int maxLabel, const int labels[]) {
    const double tiny = 0.000001;

    double amax=0;
    for(int i=0;i<n_row;i++){
        int bias = i*maxLabel;
        for(int j=0;j<maxLabel;j++){
            amax=std::max(dist[bias+j],amax);
        }
    }

    double *a = new double[n_row*maxLabel];    // dist matrix -> award matrix
    for(int i=0;i<n_row;i++) {
        int bias = i*maxLabel;
        for(int j=0;j<maxLabel;j++) {
            a[bias+j] = amax-dist[bias+j];
        }
    }

    int *labels_head = new int[maxLabel];
    int *labels_arc = new int[maxLabel];
    int *labels_next = new int[n_col];
    for(int i=0;i<maxLabel;i++)labels_head[i] = -1;
    for(int i=0;i<n_col;i++)labels_next[i] = -1;
    for(int i=0;i<n_col;i++) {
        int lb = labels[i];
        labels_next[i] = labels_head[lb];
        labels_head[lb] = i;
    }
    for(int i=0;i<maxLabel;i++)labels_arc[i] = labels_head[i];

    double* d1=new double[n_row+5];    // vertex labels, left part
    double* d2=new double[maxLabel+5];    // vertex labels, right part
    for(int i=0;i<n_row;i++)
    {
        int bias = i*maxLabel;
        d1[i]=0;
        for(int j=0;j<maxLabel;j++)
            d1[i]=std::max(d1[i],a[bias+j]);
    }
    for(int i=0;i<maxLabel;i++)
        d2[i]=0;

    int* f1=new int[maxLabel+5];    // minarg x in d1[x]+d2[i]-a[x*n_col+i]
    int* f2=new int[maxLabel+5];    // position of x in queue
    for(int i=0;i<maxLabel+5;i++)f1[i]=-1;
    for(int i=0;i<maxLabel+5;i++)f2[i]=-1;

    int N=std::max(n_row, n_col)+5;
    int (*ls)[3]=new int[N+5][3];    // vertex queue in km algorithm
    int l,r;

    int* bo1=new int[n_row+5];    // if been accessed, left part
    int* bo2=new int[maxLabel+5];    // if been accessed, right part
    for(int i=0;i<n_row+5;i++)bo1[i]=-1;
    for(int i=0;i<maxLabel+5;i++)bo2[i]=-1;

    int* p1=new int[n_row+5];    // match object, left part
    int* p2=new int[n_col+5];    // match object, right part
    for(int i=0;i<n_row+5;i++)p1[i]=-1;
    for(int i=0;i<n_col+5;i++)p2[i]=-1;

    for(int ii=0;ii<n_row;ii++)    // km algorithm, find match object for left part vertex
    {
        for(int i=0;i<maxLabel;i++)f1[i]=-1;
        for(int i=0;i<maxLabel;i++)f2[i]=-1;

        l=0; r=1; ls[r][1]=ii; ls[r][2]=0;
        while(r>0)    // loop until find the match object of ii
        {
            int flag=-1;
            while(l<r)    // queue iterate
            {
                int x;
                l++; x=ls[l][1]; bo1[x]=ii;
                for(int lb=0;lb<maxLabel;lb++)if(bo2[lb]<ii)    // search the right part
                {
                    if(abs(d1[x]+d2[lb]-a[x*maxLabel+lb])<tiny)
                    {
                        bo2[lb]=ii;
                        if(labels_arc[lb]!=-1) {
                            p2[labels_arc[lb]] = x;
                            labels_arc[lb] = labels_next[labels_arc[lb]];
                            flag = l;
                            break;
                        }
                        for(int i=labels_head[lb];i!=-1;i=labels_next[i]) {
                            r++; ls[r][0]=i; ls[r][1]=p2[i]; ls[r][2]=l;
                        }
                    }else
                    if((f1[lb]==-1)||(d1[x]+d2[lb]-a[x*maxLabel+lb]<d1[f1[lb]]+d2[lb]-a[f1[lb]*maxLabel+lb])){ f1[lb]=x; f2[lb]=l; }
                }
                if(flag!=-1)break;
            }
            if(flag!=-1){ workback(flag, ls, p2); break; }


            double dd=2147483647;
            int dd1, dd2;
            for(int lb=0;lb<maxLabel;lb++)if((bo2[lb]<ii)&&(f1[lb]>=0))    // find minist decrease
            if(d1[f1[lb]]+d2[lb]-a[f1[lb]*maxLabel+lb]<dd){
                dd = d1[f1[lb]]+d2[lb]-a[f1[lb]*maxLabel+lb];
                dd1 = f1[lb];
                dd2 = lb;
            }
            for(int i=1;i<=l;i++)if(bo1[ls[i][1]]==ii)d1[ls[i][1]]-=dd;    // update vertex label, left part
            for(int lb=0;lb<maxLabel;lb++)if(bo2[lb]==ii)d2[lb]+=dd;    // update vertex label, right part
            for(int lb=0;lb<maxLabel;lb++) {
                if((bo2[lb]<ii)&&(f1[lb]>=0)&&(abs(d1[f1[lb]]+d2[lb]-a[f1[lb]*maxLabel+lb])<tiny))
                {
                    bo2[lb]=ii;
                    if(labels_arc[lb]!=-1) {
                        p2[labels_arc[lb]] = f1[lb];
                        labels_arc[lb] = labels_next[labels_arc[lb]];
                        flag = f2[lb];
                        break;
                    }
                    for(int i=labels_head[lb];i!=-1;i=labels_next[i]) {
                        r++; ls[r][0]=i; ls[r][1]=p2[i]; ls[r][2]=f2[lb];
                    }
                }
            }
            if(flag!=-1){ workback(flag, ls, p2); break; }    // deal the augmenting path
        }
    }

    double ans=0, ans1=0, ans2=0;
    for(int i=0;i<n_row;i++)ans+=d1[i];
    for(int i=0;i<n_col;i++)ans+=d2[labels[i]];
    for(int i=0;i<n_col;i++)ans1+=a[p2[i]*maxLabel+labels[i]];
    for(int i=0;i<n_col;i++)p1[p2[i]]=i;
    for(int i=0;i<n_row;i++)ans2+=dist[i*maxLabel+labels[p1[i]]];

    std::vector<int> ret(n_row, 0);
    for(int i=0;i<n_row;i++)ret[i] = p1[i];

    delete[] a;
    delete[] d1;
    delete[] d2;
    delete[] f1;
    delete[] f2;
    delete[] bo1;
    delete[] bo2;
    delete[] p1;
    delete[] p2;
    delete[] ls;

    delete[] labels_head;
    delete[] labels_next;
    delete[] labels_arc;

    return ret;
}


//KM, python interface
std::vector<int> Optimizer::solveKM(
    const std::vector<std::vector<double>> &_dist) {

    int n_row = _dist.size();
    int n_col = _dist[0].size();

    double *dist = new double[n_row*n_col];
    for(int i=0;i<n_row;i++)
    for(int j=0;j<n_col;j++)dist[i*n_col+j]=_dist[i][j];
    std::vector<int> ret = solveKMArray(dist, n_row, n_col);

    delete[] dist;
    return ret;
}


//KM with same cost of same label, python interface
std::vector<int> Optimizer::solveKMLabel(
    const std::vector<std::vector<double>> &_dist,
    const std::vector<int> &_labels) {

    int n_row = _dist.size();
    int n_col = _labels.size();
    int maxLabel = _dist[0].size();
    int *labels = new int[n_col];
    for(int i=0;i<n_col;i++)labels[i] = _labels[i];

    double *dist = new double[n_row*maxLabel];
    for(int i=0;i<n_row;i++)
    for(int j=0;j<maxLabel;j++)dist[i*maxLabel+j]=_dist[i][j];
    std::vector<int> ret = solveKMArrayLabel(dist, n_row, n_col, maxLabel, labels);

    delete[] dist;
    delete[] labels;
    return ret;
}


//bi-graph match, change partly
std::vector<int> Optimizer::solveBiMatchChange(
const double dist[],
const int N,
const bool change[],
const int grid_asses[],
const std::string &type="km") {

    double start = clock();

    int* changeList = new int[N];
    int N2 = 0;
    for(int i=0;i<N;i++)if(change[i]){
        changeList[N2] = i;
        N2 += 1;
    }
    if(show_info)
        printf("solveBiMatchChange N: %d\n", N2);
    if(N2<N){    // only solve bi-graph match for a part of vertex
        double * new_dist = new double[N2*N2];
        #pragma omp parallel for num_threads(THREADS_NUM)
        for(int i=0;i<N2;i++){
            int gid = changeList[i];
            int bias = i*N2;
            int bias0 = gid*N;
            for(int j=0;j<N2;j++){
                int id = grid_asses[changeList[j]];
                new_dist[bias+j] = dist[bias0+id];
            }
        }

        std::vector<int> ret0(N2, 0);
        if(type=="km")
            ret0 = solveKMArray(new_dist, N2, N2);
        else
            //  ret0 = solveLapArray(new_dist, N2, std::max(50, int(0.15*N2)));
           ret0 = solveLapArray(new_dist, N2, N2);

        std::vector<int> ret(N, 0);
        for(int gid=0;gid<N;gid++)ret[gid] = grid_asses[gid];
        for(int i=0;i<N2;i++){
            int gid = changeList[i];
            int j = ret0[i];
            ret[gid] = grid_asses[changeList[j]];
        }

        delete[] new_dist;
        delete[] changeList;

        if(show_info)
            printf("solveBiMatchChange time: %.3lf\n", (clock()-start)/CLOCKS_PER_SEC);

        return ret;
    }else {    // solve all vertex
        std::vector<int> ret(N, 0);
        if(type=="km")
            ret = solveKMArray(dist, N, N);
        else
            //  ret = solveLapArray(dist, N, std::max(50, int(0.15*N)));
           ret = solveLapArray(dist, N, N);
        delete[] changeList;

        if(show_info)
            printf("solveBiMatchChange time: %.3lf\n", (clock()-start)/CLOCKS_PER_SEC);

        return ret;
    }
}

// optimize the layout by iterate of bi-graph match
std::vector<double> Optimizer::optimizeBA(
//const std::vector<int> &_ori_grid_asses,
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const std::vector<bool> &_change,
const std::string &type,
double alpha, double beta,
bool alter, const std::vector<double> alter_best,
int maxit,
const std::vector<bool> &_consider,
const std::vector<int> &_other_label) {

    double start = clock();

    // ----------------------------------preprocess step start----------------------------------------
    if(show_info)
        printf("preprocess step start\n");

    int N = _grid_asses.size();
    int num = _cluster_labels.size();
    int square_len = ceil(sqrt(N));
    int maxLabel = 0;
    for(int i=0;i<num;i++)maxLabel = std::max(maxLabel, _cluster_labels[i]+1);
    int *grid_asses = new int[N];
//    int *ori_grid_asses = new int[N];
    double (*ori_embedded)[2] = new double[N][2];
    int *cluster_labels = new int[num];
    bool *change = new bool[N];
    int *consider = new int[N];
    int *consider_e = new int[N];
    int *consider_grid = new int[N];
    for(int i=0;i<N;i++)grid_asses[i] = _grid_asses[i];
//    for(int i=0;i<N;i++)ori_grid_asses[i] = _ori_grid_asses[i];
    for(int i=0;i<N;i++) {
        ori_embedded[i][0] = _ori_embedded[i][0];
        ori_embedded[i][1] = _ori_embedded[i][1];
    }
    for(int i=0;i<num;i++)cluster_labels[i] = _cluster_labels[i];
    for(int i=0;i<N;i++)change[i] = _change[i];
    int consider_N = 0;
    for(int i=0;i<N;i++){
        consider[i] = (_consider[i]?1:0);
        consider_e[grid_asses[i]] = consider[i];
        if(_consider[i]){
            consider_grid[consider_N] = i;
            consider_N += 1;
        }
    }
    bool *is_other_label = new bool[maxLabel+1];
    for(int i=0;i<maxLabel+1;i++)is_other_label[i] = false;
    for(int i=0;i<_other_label.size();i++)is_other_label[_other_label[i]] = true;

    double *Similar_cost_matrix = new double[N*N];

    // memset(Similar_cost_matrix, 0, N*N*sizeof(double));
    
    #pragma omp parallel for num_threads(THREADS_NUM)
    for(int i1=0;i1<N;i1++){
        if((consider!=nullptr)&&(consider[i1]==0))continue;
        int bias = i1*N;
        for(int i2=0;i2<N;i2++){
            if((consider_e!=nullptr)&&(consider_e[i2]==0))continue;
            int i = bias+i2;
            Similar_cost_matrix[i] = 0;
        }
    }

    getOriginCostMatrixArrayToArray(ori_embedded, cluster_labels, Similar_cost_matrix, N, num, square_len, maxLabel, consider);

    if((type=="Triple")&&(global_cnt>=0)){    // clear the memory of measure by triples
//        printf("delete triples %d %d\n", global_N, global_hash);
        delete[] global_triples;
        delete[] global_triples_head;
        delete[] global_triples_list;
        global_cnt = -1;
        global_hash = -1;
        global_N = -1;
    }

    if(show_info)
        printf("preprocess step mid 1\n");

    double *old_cost_matrix = new double[N*N];
    double *Compact_cost_matrix = new double[N*N];
    double *old_Convex_cost_matrix = new double[N*N];
    double *Convex_cost_matrix = new double[N*N];
    double *Cn_cost_matrix = new double[N*N];
    double *new_cost_matrix = new double[N*N];
    double *other_cost_matrix = new double[N*N];

    #pragma omp parallel for num_threads(THREADS_NUM)
    for(int i1=0;i1<N;i1++){
        if((consider!=nullptr)&&(consider[i1]==0))continue;
        int bias = i1*N;
        for(int i2=0;i2<N;i2++){
            if((consider_e!=nullptr)&&(consider_e[i2]==0))continue;
            int i = bias+i2;
            old_cost_matrix[i] = 0;
            Compact_cost_matrix[i] = old_Convex_cost_matrix[i] = Convex_cost_matrix[i] = Cn_cost_matrix[i] = new_cost_matrix[i] = 0;
            other_cost_matrix[i] = 0;
        }
    }

    if(type!="Global") 
        getOtherCostMatrixArrayToArray(grid_asses, cluster_labels, other_cost_matrix, N, num, square_len, maxLabel, is_other_label, consider);

    int *ans = new int[N];
    for(int i=0;i<N;i++)ans[i] = grid_asses[i];

    double best = 2147483647;    // cost(loss) of the best ans
    double c_best = 0;    // connectivity cost(constraint) of the best ans

    std::vector<double> best_cost(4, 2147483647);    // full cost(loss) of the best ans
    double last_c_cost;    // connectivity cost(constraint) of the last ans
    std::vector<double> last_cost(4, 2147483647);    // full cost(loss) of the last ans
    std::vector<double> last_cost2(4, 2147483647);    // full cost(loss) of the last ans

    if((alter)||(beta>0))
        getCompactCostMatrixArrayToArray(grid_asses, cluster_labels, Compact_cost_matrix, N, num, square_len, maxLabel, consider);

    if(show_info)
        printf("preprocess step mid 2\n");

    int *checked = new int[N];

    double cv_time = 0;

    if(type=="CutRatio")
        last_cost = checkCostForE(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);
    else if(type=="Triple")
        last_cost = checkCostForT(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, nullptr, consider);
    else if(type=="2020")
        last_cost = checkCostFor2020(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider);
    else if(type=="Global")
        last_cost = checkCostForGlobal(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);

    if(!alter)best = last_cost[0];  //if fix alpha and beta, look for the ans with minist cost
    else best = 0;
    best_cost = last_cost;
    last_cost2 = last_cost;

    c_best = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 4, nullptr, consider_e, consider_N, consider_grid)/consider_N*N;
    if(type=="Global")c_best = 0;
    last_c_cost = c_best;

    double pre_time = (clock()-start)/CLOCKS_PER_SEC;

    // ----------------------------------preprocess step done----------------------------------------

    if(show_info)
        printf("preprocess step done\n"); 

    // ----------------------------------iterater step start----------------------------------------
    if(show_info)
        printf("iterater step start\n");

    if(show_info)
        printf("pre time %.2lf\n", pre_time);

    double a = 1;
    int downMax = 2;
    int downCnt = 0;

    double km_time = 0;
    double cm_time = 0;
    double cn_time = 0;
    double m_time = 0;
    double a_time = 0;

    bool use_knn = false;
    int k_value = 20;

    if((type!="CutRatio")&&(type!="Triple")&&(type!="2020")&&(type!="Global"))maxit = 0;

    // information of triples measure to save and load
    int *save_innerDict = new int[N*maxLabel];
    int *save_outerDict = new int[N*maxLabel*2];
    int *old_grid_asses = new int[N];
    double *T_pair = new double[2];

    int change_num = N;
    int ori_maxit = maxit;
    double avg_alpha=0, avg_beta=0;
    double ori_alpha=alpha, ori_beta=beta;

    int tot_it = 0;

    for(int it=0;it<maxit;it++) {    //iterate optimization
        tot_it = it+1;

        double start, tmp, start0, start1;
        start = clock();
        start0 = clock();
        
        if(type!="Global"){    // only adjust border
            int rand_tmp = rand()%2;
            for(int x1=0;x1<square_len;x1++) {    // grids[x1][y1]
                int bias = x1*square_len;
                for(int y1=0;y1<square_len;y1++) {
                    int gid = bias+y1;
                    change[gid] = _change[gid];
                    if(change[gid]) {
                        change[gid] = false;
                        int lb = -1;
                        if(grid_asses[gid]<num)lb = cluster_labels[grid_asses[gid]];
                        for(int xx=-1;xx<=1;xx++) {    // grids[x1+xx][y1+yy]
                            int x2 = x1+xx;
                            if((x2<0)||(x2>=square_len))continue;
                            int bias2 = x2*square_len;
                            for(int yy=-1;yy<=1;yy++) {
                                int y2 = y1+yy;
                                if((y2<0)||(y2>=square_len))continue;
                                int gid2 = bias2+y2;
                                int lb2 = -1;
                                if(grid_asses[gid2]<num)lb2 = cluster_labels[grid_asses[gid2]];
                                if(lb2!=lb) {    // different cluters, means grids[x1][y1] is in border
                                    change[gid] = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // get convexity cost matrix
        if(type=="CutRatio")
            getCostMatrixForEArrayToArray(grid_asses, cluster_labels, Convex_cost_matrix, N, num, square_len, maxLabel);
        else if(type=="Triple") {
            if((it<=0)||(change_num>=consider_N/30)) {    // too many changed grids
                getCostMatrixForTArrayToArray(grid_asses, cluster_labels, Convex_cost_matrix, N, num, square_len, maxLabel,
                true, false, save_innerDict, save_outerDict, nullptr, nullptr, consider);
            }else {    // runtime accelerate, only re-calculate changed grids from old layout
                getCostMatrixForTArrayToArray(grid_asses, cluster_labels, Convex_cost_matrix, N, num, square_len, maxLabel,
                true, true, save_innerDict, save_outerDict, old_grid_asses, T_pair, consider);
            }
        }
        else if(type=="2020")
            getCostMatrixFor2020ArrayToArray(grid_asses, cluster_labels, Convex_cost_matrix, N, num, square_len, maxLabel);

        tmp = (clock()-start)/CLOCKS_PER_SEC;
        cm_time += tmp;
        start1 = clock();

        getConnectCostMatrixArrayToArray(grid_asses, cluster_labels, Cn_cost_matrix, N, num, square_len, maxLabel, consider);

        tmp = (clock()-start1)/CLOCKS_PER_SEC;
        cn_time += tmp;

        if(alter&&(type=="Global")){    // auto adjust alpha and beta
            double dec_Similar = std::max(0.0, last_cost[1]-alter_best[0])/alter_best[2]+0.001;
            double dec_Compact = std::max(0.0, last_cost[2]-alter_best[1])/alter_best[3]+0.001;
            alpha = 0;
            beta = dec_Compact/(dec_Similar+dec_Compact);
            if(it==0)beta = ori_beta;
        }

        if((!alter)&&(type=="Global")&&(beta==1)) {
            beta = 0.99;
        }

        if(show_info) {
            printf("last cost: %.2lf %.2lf\n", last_cost[1], last_cost[2]);
            printf("alter best: %.2lf %.2lf %.2lf %.2lf\n", alter_best[0], alter_best[1], alter_best[2], alter_best[3]);
            printf("it: %d new alpha: %.2lf %.2lf\n", it, alpha, beta);
        }

        // calculate full cost matrix
        #pragma omp parallel for num_threads(THREADS_NUM)
        for(int i1=0;i1<N;i1++){
            if((consider!=nullptr)&&(consider[i1]==0))continue;
            int bias = i1*N;
            for(int i2=0;i2<N;i2++){
                if((consider!=nullptr)&&(consider_e[i2]==0))continue;

                int i = bias+i2;
                a = 1/(it+1.0);
                int lb = maxLabel;
                if(i2<num)lb = cluster_labels[i2];
                if((type!="Global")||!alter||!is_other_label[lb])
                    new_cost_matrix[i] = (1-beta-alpha)*Similar_cost_matrix[i]+beta*Compact_cost_matrix[i];
                else
                    new_cost_matrix[i] = 0.01*Similar_cost_matrix[i]+0.99*Compact_cost_matrix[i];
                old_cost_matrix[i] = new_cost_matrix[i];
                if((type!="Global")||alter) {
                    old_Convex_cost_matrix[i] = old_Convex_cost_matrix[i]*(1-a) + alpha*Convex_cost_matrix[i]*a;
                    new_cost_matrix[i] += old_Convex_cost_matrix[i];
                    if(type!="Global") {
                        new_cost_matrix[i] += Cn_cost_matrix[i]*N*(it+1)/maxit;
                        new_cost_matrix[i] += other_cost_matrix[i]*N*(it+1)/maxit;
                    }
                }
            }
        }

        tmp = (clock()-start)/CLOCKS_PER_SEC;
        m_time += tmp;

        start = clock();

        std::string b_type = "km";
        // std::string b_type = "lap";
//        if((type=="Global")&&(ori_beta+ori_alpha==0))
//            b_type = "lap";

        if(show_info)
            std::cout << "type: " << b_type << std::endl;

        std::vector<int> new_asses = solveBiMatchChange(new_cost_matrix, N, change, grid_asses, b_type);   // bi-graph match
        
        tmp = (clock()-start)/CLOCKS_PER_SEC;
        km_time += tmp;
        // printf("km time %.2lf\n", tmp);

        start = clock();

        for(int i=0;i<N;i++)old_grid_asses[i] = grid_asses[i];
        for(int i=0;i<N;i++)grid_asses[i] = new_asses[i];

        if(((alter)||(beta>0))&&((!alter)||(it<maxit-1))) {
        // if((alter)||(beta>0)) {
            if(show_info)
                printf("re calculate comp\n");
            getCompactCostMatrixArrayToArray(grid_asses, cluster_labels, Compact_cost_matrix, N, num, square_len, maxLabel, consider);    // re calculate
        }

        double cost = 2147483647;    // calculate the cost after bi-graph match
        std::vector<double> new_cost(4, 2147483647);
        if(type=="CutRatio")
            new_cost = checkCostForE(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);
        else if(type=="Triple") {
//            if(it<=1) {
            if((it<=0)||(change_num>=consider_N/30)) {
                new_cost = checkCostForT(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta,
                true, false, nullptr, T_pair, consider);
            }else {    // runtime accelerate, only re-calculate changed grids from old layout
                new_cost = checkCostForT(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta,
                true, true, old_grid_asses, T_pair, consider);
            }
        }
        else if(type=="2020")
            new_cost = checkCostFor2020(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider);
        else if(type=="Global")
            new_cost = checkCostForGlobal(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);

        if(!alter)cost = new_cost[0];
        else cost = 0;

        double c_cost = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 4, nullptr, consider_e, consider_N, consider_grid)/consider_N*N;
        if(type=="Global")c_cost = 0;

        if(alter&&(type=="Global")){
            if(show_info) {
                printf("cost1 %.2lf %.2lf\n", new_cost[1], last_cost[1]);
                printf("cost2 %.2lf %.2lf\n", new_cost[2], last_cost[2]);
            }
            if(((std::abs(new_cost[1]-last_cost2[1])+std::abs(new_cost[2]-last_cost2[2]))/consider_N<0.00015)&&(it!=1)) {
                if(show_info)
                    printf("converge 1\n");
//                break;
                maxit = it+1;
            }
            if((std::abs(new_cost[1]-last_cost[1])+std::abs(new_cost[2]-last_cost[2]))/consider_N<0.0015) {
                if(show_info)
                    printf("converge 2\n");
//                break;
                maxit = it+1;
            }
        }

        last_cost2 = last_cost;
        last_cost = new_cost;
        last_c_cost = c_cost;

//        if(((alter)&&(it==0))||(cost+c_cost<=best+c_best)) {    // update ans
        if(true) {
            best = cost;
            c_best = c_cost;
            best_cost = new_cost;
            for(int i=0;i<N;i++)ans[i] = grid_asses[i];
            downCnt = 0;
        }else downCnt += 1;

        change_num = 0;
        for(int i=0;i<N;i++)if(cluster_labels[grid_asses[i]]!=cluster_labels[old_grid_asses[i]])change_num += 1;
        if(show_info) {
            printf("cost %.2lf %.2lf %d\n", cost, c_cost, downCnt);
            printf("cost prox %.2lf comp %.2lf\n", new_cost[1], new_cost[2]);
        }

        tmp = (clock()-start)/CLOCKS_PER_SEC;
        a_time += tmp;

//        if((type=="Triple")||(type=="2020")) {   // case that need to ensure the connectivity in this function
//            if(c_best>0){
//                if(it >= maxit-1)maxit += 1;
//                if(maxit>2*ori_maxit)break;
//                continue;
//            }
//        }

        if(downCnt>=downMax)break;
    }

   // ----------------------------------iterater step done----------------------------------------

    if(show_info){
        printf("convex matrix time %.3lf\n", cm_time);
        printf("connect matrix time %.3lf\n", cn_time);
        printf("matrix time %.3lf\n", m_time);
        printf("km time %.3lf\n", km_time);
        printf("addition time %.3lf\n", a_time);
    }

    start = clock();

    std::vector<double> ret(N+6, 0);

    for(int i=0;i<N;i++)ret[i] = ans[i];
    for(int i=0;i<3;i++)ret[N+i] = best_cost[i+1];
    ret[N+3] = tot_it;
    ret[N+4] = beta;
    ret[N+5] = cv_time/1000;

    if(show_info)
        printf("end time 0 %.3lf\n", (clock()-start)/CLOCKS_PER_SEC);

    if((type=="Triple")&&(global_cnt>=0)){
//        printf("delete triples %d %d\n", global_N, global_hash);
        delete[] global_triples;
        delete[] global_triples_head;
        delete[] global_triples_list;
        global_cnt = -1;
        global_hash = -1;
        global_N = -1;
    }

    delete[] save_innerDict;
    delete[] save_outerDict;
    delete[] old_grid_asses;
    delete[] T_pair;

    delete[] checked;
    delete[] grid_asses;
    delete[] ori_embedded;
    delete[] cluster_labels;
    delete[] change;
    delete[] consider;
    delete[] consider_e;
    delete[] consider_grid;
    delete[] is_other_label;

    delete[] Similar_cost_matrix;
    delete[] Compact_cost_matrix;
    delete[] Convex_cost_matrix;
    delete[] old_Convex_cost_matrix;
    delete[] Cn_cost_matrix;
    delete[] new_cost_matrix;
    delete[] other_cost_matrix;
    delete[] ans;

    if(show_info)
        printf("end time %.3lf\n", (clock()-start)/CLOCKS_PER_SEC);
    
    return ret;
}

// optimize by search bars to swap
std::vector<double> Optimizer::optimizeSwap(
//const std::vector<int> &_ori_grid_asses,
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const std::vector<bool> &_change,
const std::string &type,
double alpha, double beta,
int maxit, int choose_k, int seed, bool innerBiMatch, int swap_cnt,
const std::vector<bool> &_consider,
const std::vector<int> &_other_label) {

    double start = clock();

    // ----------------------------------preprocess step start----------------------------------------

    int N = _grid_asses.size();
    int num = _cluster_labels.size();
    int square_len = ceil(sqrt(N));
    int maxLabel = 0;
    for(int i=0;i<num;i++)maxLabel = std::max(maxLabel, _cluster_labels[i]+1);
    int *grid_asses = new int[N];
//    int *ori_grid_asses = new int[N];
    double (*ori_embedded)[2] = new double[N][2];
    int *cluster_labels = new int[num];
    bool *change = new bool[N];
    int *change_grid = new int[N];
    int change_N = 0;
    int *consider = new int[N];
    int *consider_e = new int[N];
    int *consider_grid = new int[N];
    int consider_N = 0;
    for(int i=0;i<N;i++)grid_asses[i] = _grid_asses[i];
//    for(int i=0;i<N;i++)ori_grid_asses[i] = _ori_grid_asses[i];
    for(int i=0;i<N;i++) {
        ori_embedded[i][0] = _ori_embedded[i][0];
        ori_embedded[i][1] = _ori_embedded[i][1];
    }
    for(int i=0;i<num;i++)cluster_labels[i] = _cluster_labels[i];
    for(int i=0;i<N;i++){
        change[i] = _change[i];
        if(_change[i]){
            change_grid[change_N] = i;
            change_N += 1;
        }
    }
    for(int i=0;i<N;i++){
        consider[i] = (_consider[i]?1:0);
        consider_e[grid_asses[i]] = consider[i];
        if(_consider[i]){
            consider_grid[consider_N] = i;
            consider_N += 1;
        }
    }

    int *labels_num = new int[maxLabel];
    for(int i=0;i<maxLabel;i++)labels_num[i] = 0;
    for(int i=0;i<num;i++)labels_num[cluster_labels[i]] += 1;

    bool *is_other_label = new bool[maxLabel+1];
    for(int i=0;i<maxLabel+1;i++)is_other_label[i] = false;
    for(int i=0;i<_other_label.size();i++)is_other_label[_other_label[i]] = true;

    double pre_time = (clock()-start)/CLOCKS_PER_SEC;
    if(show_info)
        printf("pre time 0 %.2lf\n", pre_time);

    double *Similar_cost_matrix = new double[N*N];
    // memset(Similar_cost_matrix, 0, N*N*sizeof(double));
    if(type!="Edges") {
        #pragma omp parallel for num_threads(THREADS_NUM)
        for(int i1=0;i1<N;i1++){
            if((consider!=nullptr)&&(consider[i1]==0))continue;
            int bias = i1*N;
            for(int i2=0;i2<N;i2++){
                if((consider_e!=nullptr)&&(consider_e[i2]==0))continue;
                int i = bias+i2;
                Similar_cost_matrix[i] = 0;
            }
        }
        getOriginCostMatrixArrayToArray(ori_embedded, cluster_labels, Similar_cost_matrix, N, num, square_len, maxLabel, consider);
    }

    pre_time = (clock()-start)/CLOCKS_PER_SEC;
    if(show_info)
        printf("pre time 1 %.2lf\n", pre_time);

    double *Compact_cost_matrix = new double[N*N];
    // memset(Compact_cost_matrix, 0, N*N*sizeof(double));
    if(type!="Edges") {
        #pragma omp parallel for num_threads(THREADS_NUM)
        for(int i1=0;i1<N;i1++){
            if((consider!=nullptr)&&(consider[i1]==0))continue;
            int bias = i1*N;
            for(int i2=0;i2<N;i2++){
                if((consider_e!=nullptr)&&(consider_e[i2]==0))continue;
                int i = bias+i2;
                Compact_cost_matrix[i] = 0;
            }
        }
        getCompactCostMatrixArrayToArray(grid_asses, cluster_labels, Compact_cost_matrix, N, num, square_len, maxLabel, consider);
    }

    pre_time = (clock()-start)/CLOCKS_PER_SEC;
    if(show_info)
        printf("pre time 2 %.2lf\n", pre_time);

    double *other_cost_matrix = new double[N*N];
    // memset(other_cost_matrix, 0, N*N*sizeof(double));
    if(type!="Edges") {
        #pragma omp parallel for num_threads(THREADS_NUM)
        for(int i1=0;i1<N;i1++){
            if((consider!=nullptr)&&(consider[i1]==0))continue;
            int bias = i1*N;
            for(int i2=0;i2<N;i2++){
                if((consider_e!=nullptr)&&(consider_e[i2]==0))continue;
                int i = bias+i2;
                other_cost_matrix[i] = 0;
            }
        }
        getOtherCostMatrixArrayToArray(grid_asses, cluster_labels, other_cost_matrix, N, num, square_len, maxLabel, is_other_label, consider);
    }

    pre_time = (clock()-start)/CLOCKS_PER_SEC;
    if(show_info)
        printf("pre time 3 %.2lf\n", pre_time);

    int *ans = new int[N];
    for(int i=0;i<N;i++)ans[i] = grid_asses[i];
    double best = 2147483647;    // cost of ans
    double c_best = 0;    // connectivity cost(constraint) of ans

    double *old_T_pair = new double[2];    // convexity of triples
    double *old_D_pair = new double[N*N*2];
    int *old_grid_asses = new int[N];
    for(int i=0;i<N;i++)old_grid_asses[i] = grid_asses[i];

    bool *if_disconn = new bool[N];   // disconnect of now grids
    for(int i=0;i<N;i++)if_disconn[i] = false;
    bool *if_disconn2 = new bool[N];   // disconnect of now grids
    for(int i=0;i<N;i++)if_disconn2[i] = false;
    int *checked = new int[N];

    if(type=="CutRatio")
        best = checkCostForE(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider)[0];
    else if(type=="AreaRatio")
        best = checkCostForS(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider)[0];
    else if(type=="PerimeterRatio")
        best = checkCostForC(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider, consider_N, consider_grid)[0];
    else if(type=="2020")
        best = checkCostFor2020(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider)[0];
    else if(type=="TB")
        best = checkCostForTB(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider)[0];
    else if(type=="Triple")
        best = checkCostForT(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, false, old_grid_asses, old_T_pair, consider)[0];
    else if(type=="Global")
        best = checkCostForGlobal(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider)[0];
    else if(type=="T2")
        best = checkCostForT2(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, false, old_grid_asses, old_T_pair, old_D_pair, consider)[0];
    else if(type=="Edges")
        best = 0;
    if(type!="Edges")
        c_best = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, if_disconn, consider_e, consider_N, consider_grid)/consider_N*N;
    // c_best = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, if_disconn)/consider_N*N;
    else c_best = 0;
    // ----------------------------------preprocess step done----------------------------------------

    pre_time = (clock()-start)/CLOCKS_PER_SEC;
    if(show_info)
        printf("pre time %.2lf\n", pre_time);

    // ----------------------------------swap step start----------------------------------------

    int downMax = 3;
    int downCnt = 0;
    srand(seed);

    int check_turns1 = 0;
    int check_turns2 = 0;
    int once_flag = 1;
    double s_time = 0;
    double sp_time = 0;
    double a_time = 0;

    bool use_knn = false;
    int k_value = 50;

    int max_num = 2;    // max bar length to search
    for(int it=0;it<maxit;it++){    //枚举轮次
        if(type=="CutRatio"||type=="AreaRatio"||type=="PerimeterRatio"||type=="2020"||type=="Triple"||type=="T2"||type=="TB"||type=="Edges"){

            if(swap_cnt<=0) {
                break;
            }

            int update_flag = 0;    // if layout been updated
            int improve_flag = 0;    // if find a better layout
            // if(type=="AreaRatio")max_num = 1;
            // if(type=="Triple")max_num = 1;
            // if(type=="2020")max_num = 1;
            // if(type=="PerimeterRatio")max_num = 6;
            if(type=="Edges")max_num = 6;

            double (*label_pairs)[2] = new double[maxLabel][2];    // convexity of every cluster
            int *worst_gid = new int[2*N*max_num];    // bar2
            int worst_cnt = 0;    // number of bar2
            int *order1 = new int[2*N];    // enumerate order of bar1
            int *order2 = new int[2*N];    // enumerate order of bar2
            int *E_grid = new int[N];    // count of edges with a different clusters of every grid
            int *E_grid2 = new int[N];    // count of edges with the bar1 clusters information of every grid
            int *labels_cnt = new int[maxLabel+1];    // count of edges with every clusters, bar1
            int *labels_cnt2 = new int[maxLabel+1];    // count of edges with every clusters, bar2
            int check_cnt = 1;
            for(int i=0;i<N;i++)checked[i]=0;

            double *all_cost = new double[2*N];    // cost of swapping
            double *all_c_cost = new double[2*N];    // connectivity cost(constraint) of swapping
            double *all_b_cost = new double[2*N];    // edges with blank grids
            double *all_o_cost = new double[2*N];    
            double *all_e_cost = new double[2*N];    
            double *all_dec = new double[2*N];    // edges with blank grids

            double start = clock();

            for(int now_num=max_num;now_num>=1;now_num--){    // long bar first

                if(swap_cnt<=0) {
                    break;
                }

                for(int i=0;i<N;i++){
                    order1[i] = i;
                }

                std::random_shuffle(order1, order1+N);    //shuffle the order

                for(int gid1_o=0;gid1_o<N;gid1_o++){    //enumerate bar1

                    if(swap_cnt<=0) {
                        break;
                    }

                    int gid1 = order1[gid1_o];
                    if(!change[gid1])continue;
                    int id1 = grid_asses[gid1];
                    if(id1>=num)continue;
                    int mainLabel1 = cluster_labels[id1];
                    int x1 = gid1/square_len;
                    int y1 = gid1%square_len;
                    for(int ori=0;ori<2;ori++){
                        if((ori>0)&&(now_num==1))continue;
                        int ori_gid1[100];
                        if((ori==0)&&(x1>=now_num-1)){    //V bar
                            for(int i=0;i<now_num;i++)ori_gid1[i] = gid1 - i*square_len;
                        }else
                        if((ori==1)&&(y1>=now_num-1)){    //H bar
                            for(int i=0;i<now_num;i++)ori_gid1[i] = gid1 - i;
                        }else continue;

                        for(int i=0;i<maxLabel+1;i++)labels_cnt[i] = 0;
                        for(int i=0;i<maxLabel+1;i++)labels_cnt2[i] = 0;

                        int flag=0;
                        int dc_flag=0;

                        for(int i=0;i<now_num;i++){
                            int gid = ori_gid1[i];
                            int id = grid_asses[gid];
                            if((id>=num)||(cluster_labels[id]!=cluster_labels[id1])){
                                flag = 1;
                                break;
                            }
                            if(!change[gid]){
                                flag = 1;
                                break;
                            }
                            checkEdgeSingleForLabel(labels_cnt, gid, grid_asses, cluster_labels, N, num, square_len, maxLabel);
                        }
                        if(flag>0)continue;    //illegal bar

                        for(int i=0;i<now_num;i++){
                            int gid = ori_gid1[i];
                            dc_flag += if_disconn[gid];
                        }

                        int dcnt = 0, mainLabel2 = maxLabel;
                        for(int i=0;i<maxLabel+1;i++){
                            if(i==cluster_labels[id1])continue;
                            dcnt += labels_cnt[i];
                            if(labels_cnt[i]>labels_cnt[mainLabel2])mainLabel2 = i;
                        }
                        if(dcnt<=now_num)continue;    //not enough edges with a different clusters

                        // if((type=="Triple")&&(dcnt<=now_num+1))continue;

                        check_turns1 += 1;

                        checkEdgeArrayForSingleLabel(E_grid, mainLabel1, grid_asses, cluster_labels, N, num, square_len, maxLabel, consider, consider_N, consider_grid);
                        checkEdgeArray(E_grid2, grid_asses, cluster_labels, N, num, square_len, maxLabel, consider, consider_N, consider_grid);
                        worst_cnt = 0;

                        // for(int gid2=0;gid2<N;gid2++){    // enumerate bar2
                        for(int idx=0, gid2=change_grid[0]; idx<change_N; idx++, gid2=change_grid[idx]){    // enumerate bar2
                            int id2 = grid_asses[gid2];
                            int lb2 = maxLabel;
                            if(id2<num)lb2 = cluster_labels[id2];
                            if((id2<num)&&(labels_cnt[lb2]==0))continue;   // cluster that have no edge with bar1
                            int x2 = gid2/square_len;
                            int y2 = gid2%square_len;
                            for(int ori=0;ori<2;ori++){
                                if((ori>0)&&(now_num==1))continue;
                                int now_gid2[100];
                                if((ori==0)&&(x2>=now_num-1)){    //V bar
                                    for(int i=0;i<now_num;i++)now_gid2[i] = gid2 - i*square_len;
                                }else
                                if((ori==1)&&(y2>=now_num-1)){    //H bar
                                    for(int i=0;i<now_num;i++)now_gid2[i] = gid2 - i;
                                }else continue;

                                int flag=0;
                                int dcnt=0;
                                int dcnt2=0;
                                for(int i=0;i<now_num;i++){
                                    int gid = now_gid2[i];
                                    int id = grid_asses[gid];
                                    int lb = maxLabel;
                                    if(id<num)lb = cluster_labels[id];
                                    if(lb!=lb2){
                                        flag = 1;
                                        break;
                                    }
                                    if(!change[gid]){
                                        flag = 1;
                                        break;
                                    }
                                    dcnt += E_grid[gid];
                                    dcnt2 += E_grid2[gid];
                                }

                                if(flag>0)continue;
                                if((dc_flag==0)&&(labels_num[mainLabel1]>1)) {    // bar1 is connective with cluster
                                    if(dcnt<=std::max(now_num-2,0))continue;    // not enough edges with the bar1 cluster
                                    if(dcnt2<=now_num)continue;    // not enough edges with a different cluster
                                }else {
                                    if(dcnt2<=std::max(now_num-1,0))continue;    // not enough edges with a different cluster
                                }

                                // if((type=="Triple")&&(x1!=x2)&&(y1!=y2))continue;

                                for(int i=0;i<now_num;i++)worst_gid[worst_cnt*now_num+i] = now_gid2[i];    // be candidate
                                worst_cnt += 1;
                            }
                        }

                        if(worst_cnt==0)continue;

                        for(int i=0;i<worst_cnt;i++){
                            order2[i] = i;
                        }
                        // std::random_shuffle(order2, order2+worst_cnt);

                        double now_best = 0;    // cost of swapping with best bar2
                        double now_c_best = 0;    // connectivity cost(constraint) of swapping with best bar2
                        double now_b_best = 0;    // edges with blank grids
                        double now_o_best = 0;
                        double now_e_best = 0;   // edges with other clusters
                        double ori_cost = 0;    // cost
                        double ori_c_cost = 0;    // connectivity cost(constraint)

                        if(type=="CutRatio")
                            now_best =checkCostForE(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider)[0];
                        else if(type=="AreaRatio")
                            now_best =checkCostForS(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, false, label_pairs, -1, -1, consider)[0];
                        else if(type=="PerimeterRatio")
                            now_best =checkCostForC(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, false, label_pairs, -1, -1, consider, consider_N, consider_grid)[0];
                        else if(type=="2020")
                            now_best =checkCostFor2020(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, false, label_pairs, -1, -1, consider)[0];
                        else if(type=="TB")
                            now_best =checkCostForTB(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, false, label_pairs, -1, -1, consider)[0];
                        else if(type=="Triple") {
                            now_best =checkCostForT(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, true, old_grid_asses, old_T_pair, consider)[0];
                            for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = grid_asses[tmp_gid];
                        }
                        else if(type=="T2") {
                            now_best =checkCostForT2(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, true, old_grid_asses, old_T_pair, old_D_pair, consider)[0];
                            for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = grid_asses[tmp_gid];
                        }

                        if(type!="Edges")
                            now_c_best = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, nullptr, consider_e, consider_N, consider_grid)/consider_N*N;
                            // now_c_best = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, nullptr)/consider_N*N;

                        ori_cost = now_best;
                        ori_c_cost = now_c_best;

                        int best_gid = -1;    // best bar2

//                        printf("worst cnt %d\n", worst_cnt);

                        double tmp_start = clock();

                        // int tmp_th = std::min(THREADS_NUM, worst_cnt);
                        #pragma omp parallel for num_threads(std::min(worst_cnt, THREADS_NUM))
                        for(int jj=0;jj<worst_cnt;jj++){    // enumerate candidate bar2
                            check_turns2 += 1;
                            int j = order2[jj];

                            all_dec[j] = -1;

                            int flag = 0;
                            for(int k1=0;k1<now_num;k1++)
                            for(int k2=0;k2<now_num;k2++){
                                if(ori_gid1[k1]==worst_gid[j*now_num+k2]){
                                    flag += 1;
                                }
                            }
                            if(flag>0)continue;    // have share grid

                            int id1 = grid_asses[ori_gid1[0]];
                            int id2 = grid_asses[worst_gid[j*now_num]];
                            int lb1 = cluster_labels[id1];
                            int lb2 = maxLabel;
                            if(id2<num)lb2 = cluster_labels[id2];
                            if(lb1==lb2)continue;    // same cluster

                            int *tmp_grid_asses = new int[N];
                            int *tmp_checked = new int[N];
                            memcpy(tmp_grid_asses, grid_asses, N*sizeof(int));
//                            int *tmp_grid_asses = grid_asses;
//                            int *tmp_checked = checked;

                            for(int k=0;k<now_num;k++){    //swap
                                std::swap(tmp_grid_asses[ori_gid1[k]], tmp_grid_asses[worst_gid[j*now_num+k]]);
                            }

                            double cost = 0;    // new cost
                            double c_cost = 0;    // new connectivity cost
                            double b_cost = 0;
                            double o_cost = 0;
                            double e_cost = 0;

                            if(type=="CutRatio")
                                cost = checkCostForE(Similar_cost_matrix, Compact_cost_matrix, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider)[0];
                            else if(type=="AreaRatio")
                                cost = checkCostForS(Similar_cost_matrix, Compact_cost_matrix, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, true, label_pairs, lb1, lb2, consider)[0];
                            else if(type=="PerimeterRatio")
                                cost = checkCostForC(Similar_cost_matrix, Compact_cost_matrix, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, true, label_pairs, lb1, lb2, consider, consider_N, consider_grid)[0];
                            else if(type=="2020")
                                cost = checkCostFor2020(Similar_cost_matrix, Compact_cost_matrix, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, true, label_pairs, lb1, lb2, consider)[0];
                            else if(type=="TB")
                                cost = checkCostForTB(Similar_cost_matrix, Compact_cost_matrix, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, true, label_pairs, lb1, lb2, consider)[0];
                            else if(type=="Triple")
                                cost = checkCostForT(Similar_cost_matrix, Compact_cost_matrix, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, true, old_grid_asses, old_T_pair, consider)[0];
                            else if(type=="T2")
                                cost = checkCostForT2(Similar_cost_matrix, Compact_cost_matrix, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, true, old_grid_asses, old_T_pair, old_D_pair, consider)[0];

                            if(type!="Edges")
                                c_cost = N*checkConnectForAll(tmp_grid_asses, cluster_labels, tmp_checked, N, num, square_len, maxLabel, 8, if_disconn2, consider_e, consider_N, consider_grid)/consider_N*N;

                            // c_cost = N*checkConnectForAll(tmp_grid_asses, cluster_labels, tmp_checked, N, num, square_len, maxLabel, 8, nullptr)/consider_N*N;
                            int disconn_flag = false;
                            for(int k=0;k<now_num;k++){
                                if(if_disconn[ori_gid1[k]])disconn_flag = true;
                                if(if_disconn[worst_gid[j*now_num+k]])disconn_flag = true;
                                if(if_disconn2[ori_gid1[k]])disconn_flag = true;
                                if(if_disconn2[worst_gid[j*now_num+k]])disconn_flag = true;
                            }
                            if(checkDisconnectChange(N, if_disconn, if_disconn2, consider, consider_N, consider_grid))disconn_flag = true;
                            if(!disconn_flag)c_cost = ori_c_cost;

                            for(int k=0;k<now_num;k++){
                                b_cost -= 0.000001*checkBlankForGrid(ori_gid1[k], tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel);
                            }
                            for(int k=0;k<now_num;k++){
                                b_cost -= 0.000001*checkBlankForGrid(worst_gid[j*now_num+k], tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel);
                            }

                            if(type!="Edges") {
                                for(int k=0;k<now_num;k++){
                                    int gid = ori_gid1[k];
                                    o_cost += N*other_cost_matrix[gid*N+tmp_grid_asses[gid]];
                                }
                                for(int k=0;k<now_num;k++){
                                    int gid = worst_gid[j*now_num+k];
                                    o_cost += N*other_cost_matrix[gid*N+tmp_grid_asses[gid]];    
                                }   
                            }

                            if(type=="Edges") {
                                for(int k=0;k<now_num;k++){
                                    int gid = ori_gid1[k];
                                    e_cost += checkEdgeSingle(gid, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel);
                                }
                                for(int k=0;k<now_num;k++){
                                    int gid = worst_gid[j*now_num+k];
                                    e_cost += checkEdgeSingle(gid, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel);    
                                }  
                            }                         

                            for(int k=0;k<now_num;k++){    //swap back
                                std::swap(tmp_grid_asses[ori_gid1[k]], tmp_grid_asses[worst_gid[j*now_num+k]]);
                            }

                            for(int k=0;k<now_num;k++){
                                b_cost += 0.000001*checkBlankForGrid(ori_gid1[k], tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel);
                            }
                            for(int k=0;k<now_num;k++){
                                b_cost += 0.000001*checkBlankForGrid(worst_gid[j*now_num+k], tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel);
                            }

                            if(type!="Edges") {
                                for(int k=0;k<now_num;k++){
                                    int gid = ori_gid1[k];
                                    o_cost -= N*other_cost_matrix[gid*N+tmp_grid_asses[gid]];
                                }
                                for(int k=0;k<now_num;k++){
                                    int gid = worst_gid[j*now_num+k];
                                    o_cost -= N*other_cost_matrix[gid*N+tmp_grid_asses[gid]];    
                                } 
                            } 

                            if(type=="Edges") {
                                for(int k=0;k<now_num;k++){
                                    int gid = ori_gid1[k];
                                    e_cost -= checkEdgeSingle(gid, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel);
                                }
                                for(int k=0;k<now_num;k++){
                                    int gid = worst_gid[j*now_num+k];
                                    e_cost -= checkEdgeSingle(gid, tmp_grid_asses, cluster_labels, N, num, square_len, maxLabel);    
                                }  
                            }  

                            all_cost[j] = cost;
                            all_c_cost[j] = c_cost;
                            all_b_cost[j] = b_cost;
                            all_o_cost[j] = o_cost;
                            all_e_cost[j] = e_cost;
                            all_dec[j] = ori_cost + ori_c_cost - cost - c_cost - b_cost - o_cost - e_cost;

                            // printf("swap %d %d %d\n", now_num, ori_gid1[0], worst_gid[j*now_num+0]);
                            // printf("cost %.2lf %.2lf %.2lf %.2lf %.2lf\n", ori_cost, ori_c_cost, cost, c_cost, b_cost, o_cost);

                            delete[] tmp_grid_asses;
                            delete[] tmp_checked;
                        }

                        for(int jj=0;jj<worst_cnt;jj++){    // enumerate candidate bar2

                            check_turns2 += 1;
                            int j = order2[jj];

                            double cost = all_cost[j];
                            double c_cost = all_c_cost[j];
                            double b_cost = all_b_cost[j];
                            double o_cost = all_o_cost[j];
                            double e_cost = all_e_cost[j];

                            if(ori_cost + ori_c_cost - all_dec[j]<now_best+now_c_best+now_b_best+now_o_best+now_e_best){
                                now_best = cost;
                                now_c_best = c_cost;
                                now_b_best = b_cost;
                                now_o_best = o_cost;
                                now_e_best = e_cost;
                                best_gid = j;
                            }
                        }

                        sp_time += (clock()-tmp_start)/CLOCKS_PER_SEC;

                        if(best_gid!=-1){    // choose the best bar2 to swap

//                            best_gid = soft_choose(all_dec, worst_cnt);
                            best_gid = best_k_choose(all_dec, worst_cnt, choose_k);

                            if(swap_cnt<=0) {
                                break;
                            }

                            // printf("swap %d %d %d\n", now_num, ori_gid1[0], worst_gid[best_gid*now_num+0]);
                            // printf("cost %.2lf %.2lf %.2lf %.2lf %.2lf\n", ori_cost, ori_c_cost, all_cost[best_gid], all_c_cost[best_gid], all_b_cost[best_gid], all_o_cost[best_gid]);

                            update_flag = 1;

                            double cost = now_best;
                            double c_cost = now_c_best;

                            for(int k=0;k<now_num;k++){
                                std::swap(grid_asses[ori_gid1[k]], grid_asses[worst_gid[best_gid*now_num+k]]);
                            }

                            if(type=="CutRatio")
                                cost = checkCostForE(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider)[0];
                            else if(type=="AreaRatio")
                                cost = checkCostForS(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider)[0];
                            else if(type=="PerimeterRatio")
                                cost = checkCostForC(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider, consider_N, consider_grid)[0];
                            else if(type=="2020")
                                cost = checkCostFor2020(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider)[0];
                            else if(type=="TB")
                                cost = checkCostForTB(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider)[0];
                            else if(type=="Triple") {
                                cost = checkCostForT(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, true, old_grid_asses, old_T_pair, consider)[0];
                                for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = grid_asses[tmp_gid];
                            }
                            else if(type=="T2") {
                                cost = checkCostForT2(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, true, old_grid_asses, old_T_pair, old_D_pair, consider)[0];
                                for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = grid_asses[tmp_gid];
                            }

                            if(type!="Edges")
                                c_cost = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, if_disconn, consider_e, consider_N, consider_grid);  // update disconnect
                            // c_cost = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, if_disconn);  // update disconnect
                            
                            swap_cnt -= 1;

                            best = cost;
                            c_best = c_cost;

                            for(int k=0;k<N;k++)ans[k] = grid_asses[k];
                            improve_flag = 1;
                        }
                    }
                }
            }

            delete[] label_pairs;
            delete[] E_grid;
            delete[] E_grid2;
            delete[] worst_gid;
            delete[] order1;
            delete[] order2;
            delete[] labels_cnt;
            delete[] labels_cnt2;

            delete[] all_cost;
            delete[] all_c_cost;
            delete[] all_b_cost;
            delete[] all_o_cost;
            delete[] all_e_cost;
            delete[] all_dec;

            double tmp = (clock()-start)/CLOCKS_PER_SEC;
            s_time += tmp;

            start = clock();

            max_num += 2;

            if(improve_flag==1){
                downCnt=0;
            }else downCnt += 1;

            tmp = (clock()-start)/CLOCKS_PER_SEC;
            a_time += tmp;

            if(show_info)
                printf("downCnt %d\n", downCnt);
            if(downCnt>=downMax) {
//                if(type=="PerimeterRatio")continue;
                break;
            }
        }
    }

    // ----------------------------------swap step done----------------------------------------

    double swap_time = (clock()-start)/CLOCKS_PER_SEC;
    if(show_info)
        printf("swap time %.2lf\n", swap_time);

    if(innerBiMatch) {
        double *cost_matrix = new double[N*N];    //bi-graph match in each clusters to ensure the minist Similar cost

        #pragma omp parallel for num_threads(THREADS_NUM)
        for(int gid2=0;gid2<N;gid2++){
            if((consider!=nullptr)&&(consider[gid2]==0))continue;
            int lb2 = -1;
            int id2 = ans[gid2];
            if(id2<num)lb2 = cluster_labels[id2];

            for(int gid1=0;gid1<N;gid1++){
                if((consider!=nullptr)&&(consider[gid1]==0))continue;
                int lb1 = -1;
                int id1 = ans[gid1];
                if(id1<num)lb1 = cluster_labels[id1];
                if(lb1==lb2){
                    cost_matrix[gid2*N+id1] = Similar_cost_matrix[gid2*N+id1];
                }else {
                    cost_matrix[gid2*N+id1] = N;
                }
            }
        }

        std::vector<int> new_grid_asses = solveBiMatchChange(cost_matrix, N, change, ans);

        delete[] cost_matrix;
        for(int i=0;i<N;i++)ans[i] = new_grid_asses[i];
    }

    // printf("final cost\n");
    std::vector<double> cost(4, -1);
    if(type=="CutRatio")
        cost = checkCostForE(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);
    else if(type=="AreaRatio")
        cost = checkCostForS(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider);
    else if(type=="PerimeterRatio")
        cost = checkCostForC(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider, consider_N, consider_grid);
    else if(type=="2020")
        cost = checkCostFor2020(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider);
    else if(type=="TB")
        cost = checkCostForTB(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider);
    else if(type=="Triple") {
        cost = checkCostForT(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, true, old_grid_asses, old_T_pair, consider);
        for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = ans[tmp_gid];
    }
    else if(type=="T2") {
        cost = checkCostForT2(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, true, old_grid_asses, old_T_pair, old_D_pair, consider);
        for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = ans[tmp_gid];
    }
    else if(type=="Global")
        cost = checkCostForGlobal(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);
    // printf("cost %.6lf %.6lf %.6lf\n", cost[1], cost[2], cost[3]);

    std::vector<double> ret(N+3, 0);
    for(int i=0;i<N;i++)ret[i] = ans[i];
    for(int i=0;i<3;i++)ret[N+i] = cost[i+1];

    delete[] checked;
    delete[] if_disconn;
    delete[] if_disconn2;

    delete[] old_grid_asses;
    delete[] old_T_pair;
    delete[] old_D_pair;

    delete[] grid_asses;
    delete[] ori_embedded;
    delete[] cluster_labels;
    delete[] change;
    delete[] change_grid;
    delete[] consider;
    delete[] consider_e;
    delete[] consider_grid;
    delete[] labels_num;
    delete[] is_other_label;

    delete[] Similar_cost_matrix;
    delete[] Compact_cost_matrix;
    delete[] other_cost_matrix;
    delete[] ans;

    double f_time = (clock()-start)/CLOCKS_PER_SEC;
    if(show_info)
        printf("final time %.2lf\n", f_time);

    return ret;
}


std::vector<int> changeLabelsForConvexity(
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const std::vector<bool> &_change,
const std::string &type, 
const std::vector<bool> &_consider) {

    Optimizer op(0);
    double start = clock();

    // ----------------------------------preprocess step start----------------------------------------

    int N = _grid_asses.size();
    int num = _cluster_labels.size();
    int square_len = ceil(sqrt(N));
    int maxLabel = 0;
    int considerMaxLabel = 0;
    for(int i=0;i<num;i++)maxLabel = std::max(maxLabel, _cluster_labels[i]+1);
    int *grid_asses = new int[N];
//    int *ori_grid_asses = new int[N];
    int *cluster_labels = new int[num];
    bool *change = new bool[N];
    int *change_grid = new int[N];
    int change_N = 0;
    int *consider = new int[N];
    int *consider_e = new int[N];
    int *consider_grid = new int[N];
    int consider_N = 0;
    for(int i=0;i<N;i++)grid_asses[i] = _grid_asses[i];
    for(int i=0;i<num;i++)cluster_labels[i] = _cluster_labels[i];
    
    for(int x1=0;x1<square_len;x1++) {    // grids[x1][y1]
        int bias = x1*square_len;
        for(int y1=0;y1<square_len;y1++) {
            int gid = bias+y1;
            change[gid] = _change[gid];
            if(change[gid]) {
                change[gid] = false;
                int lb = -1;
                if(grid_asses[gid]<num)lb = cluster_labels[grid_asses[gid]];
                for(int xx=-1;xx<=1;xx++) {    // grids[x1+xx][y1+yy]
                    int x2 = x1+xx;
                    if((x2<0)||(x2>=square_len))continue;
                    int bias2 = x2*square_len;
                    for(int yy=-1;yy<=1;yy++) {
                        int y2 = y1+yy;
                        if((y2<0)||(y2>=square_len))continue;
                        int gid2 = bias2+y2;
                        int lb2 = -1;
                        if(grid_asses[gid2]<num)lb2 = cluster_labels[grid_asses[gid2]];
                        if(lb2!=lb) {    // different cluters, means grids[x1][y1] is in border
                            change[gid] = true;
                            break;
                        }
                    }
                }
            }
            if(change[gid]){
                change_grid[change_N] = gid;
                change_N += 1;
            }
        }   
    }

    for(int i=0;i<N;i++){
        consider[i] = (_consider[i]?1:0);
        consider_e[grid_asses[i]] = consider[i];
        if(_consider[i]){
            consider_grid[consider_N] = i;
            consider_N += 1;
        }
    }
    for(int i=0;i<num;i++)if(consider_e[i])
    considerMaxLabel = std::max(considerMaxLabel, _cluster_labels[i]+1);

    double pre_time = (clock()-start)/CLOCKS_PER_SEC;

    double *Similar_cost_matrix = new double[N*N];
    #pragma omp parallel for num_threads(THREADS_NUM)
    for(int i1=0;i1<N;i1++){
        if((consider!=nullptr)&&(consider[i1]==0))continue;
        int bias = i1*N;
        for(int i2=0;i2<N;i2++){
            if((consider_e!=nullptr)&&(consider_e[i2]==0))continue;
            int i = bias+i2;
            Similar_cost_matrix[i] = 0;
        }
    }

    double *Compact_cost_matrix = new double[N*N];
    #pragma omp parallel for num_threads(THREADS_NUM)
    for(int i1=0;i1<N;i1++){
        if((consider!=nullptr)&&(consider[i1]==0))continue;
        int bias = i1*N;
        for(int i2=0;i2<N;i2++){
            if((consider_e!=nullptr)&&(consider_e[i2]==0))continue;
            int i = bias+i2;
            Compact_cost_matrix[i] = 0;
        }
    }

    int *ans = new int[N];
    for(int i=0;i<N;i++)ans[i] = grid_asses[i];
    double best = 2147483647;    // cost of ans
    double c_best = 0;    // connectivity cost(constraint) of ans

    int *old_grid_asses = new int[N];
    for(int i=0;i<N;i++)old_grid_asses[i] = grid_asses[i];

    int *checked = new int[N];

    if(type=="PerimeterRatio")
        best = op.checkCostForC(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, 1, 0, false, false, nullptr, -1, -1, consider, consider_N, consider_grid)[0];
    
    c_best = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, nullptr, consider_e, consider_N, consider_grid)/consider_N*N;


    int *label_change_cnt = new int[maxLabel];
    for(int i=0;i<maxLabel;i++)label_change_cnt[i] = 0;
    // ----------------------------------preprocess step done----------------------------------------

    // ----------------------------------swap step start----------------------------------------

    int downMax = 3;
    int downCnt = 0;
    srand(10);

    int once_flag = 1;

    bool use_knn = false;
    int k_value = 50;

    int max_num = 3;    // max bar length to search
    for(int it=0;it<2;it++){    //枚举轮次
        if(type=="PerimeterRatio"){

            int update_flag = 0;
            int improve_flag = 0;

            double (*label_pairs)[2] = new double[maxLabel][2];    // convexity of every cluster
            int *order1 = new int[2*N];    // enumerate order of bar1
            int *labels_cnt = new int[maxLabel+1];    // count of edges with every clusters, bar1

            double start = clock();

            for(int now_num=max_num;now_num>=1;now_num--){    // long bar first

                for(int i=0;i<N;i++){
                    order1[i] = i;
                }

                std::random_shuffle(order1, order1+N);    //shuffle the order

                for(int gid1_o=0;gid1_o<N;gid1_o++){    //enumerate bar1

                    int gid1 = order1[gid1_o];
                    if(!change[gid1])continue;
                    int id1 = grid_asses[gid1];
                    if(id1>=num)continue;
                    int mainLabel1 = cluster_labels[id1];
                    int x1 = gid1/square_len;
                    int y1 = gid1%square_len;
                    for(int ori=0;ori<2;ori++){
                        if((ori>0)&&(now_num==1))continue;
                        int ori_gid1[100];
                        if((ori==0)&&(x1>=now_num-1)){    //V bar
                            for(int i=0;i<now_num;i++)ori_gid1[i] = gid1 - i*square_len;
                        }else
                        if((ori==1)&&(y1>=now_num-1)){    //H bar
                            for(int i=0;i<now_num;i++)ori_gid1[i] = gid1 - i;
                        }else continue;

                        for(int i=0;i<maxLabel+1;i++)labels_cnt[i] = 0;

                        int flag=0;

                        for(int i=0;i<now_num;i++){
                            int gid = ori_gid1[i];
                            int id = grid_asses[gid];
                            if((id>=num)||(cluster_labels[id]!=cluster_labels[id1])){
                                flag = 1;
                                break;
                            }
                            if(!change[gid]){
                                flag = 1;
                                break;
                            }
                            checkEdgeSingleForLabel(labels_cnt, gid, grid_asses, cluster_labels, N, num, square_len, maxLabel);
                        }
                        if(flag>0)continue;    //illegal bar

                        int dcnt = 0, mainLabel2 = maxLabel;
                        for(int i=0;i<maxLabel+1;i++){
                            if(i==cluster_labels[id1])continue;
                            dcnt += labels_cnt[i];
                            if(labels_cnt[i]>labels_cnt[mainLabel2])mainLabel2 = i;
                        }
                        if(mainLabel2>=considerMaxLabel)continue;
                        if(dcnt<=now_num)continue;    //not enough edges with a different clusters

                        // if(label_change_cnt[mainLabel1]-now_num<-6)continue;
                        // if(label_change_cnt[mainLabel2]+now_num>6)continue;


                        double now_best = 0;    // cost of swapping with best bar2
                        double now_c_best = 0;    // connectivity cost(constraint) of swapping with best bar2
                        double ori_cost = 0;    // cost
                        double ori_c_cost = 0;    // connectivity cost(constraint)

                        if(type=="PerimeterRatio")
                            now_best = op.checkCostForC(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, 1, 0, true, false, label_pairs, -1, -1, consider, consider_N, consider_grid)[0];

                        now_c_best = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, nullptr, consider_e, consider_N, consider_grid)/consider_N*N;

                        ori_cost = now_best;
                        ori_c_cost = now_c_best;

                        double tmp_start = clock();

                        int lb1 = mainLabel1;
                        int lb2 = mainLabel2;

                        for(int k=0;k<now_num;k++){    //change
                            cluster_labels[grid_asses[ori_gid1[k]]] = lb2;
                        }

                        double cost = 0;    // new cost
                        double c_cost = 0;    // new connectivity cost

                        if(type=="PerimeterRatio")
                            cost = op.checkCostForC(Similar_cost_matrix, Compact_cost_matrix, grid_asses, cluster_labels, N, num, square_len, maxLabel, 1, 0, false, true, label_pairs, lb1, lb2, consider, consider_N, consider_grid)[0];
    
                        c_cost = N*checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, nullptr, consider_e, consider_N, consider_grid)/consider_N*N;                       

                        if(cost+c_cost>=ori_cost+ori_c_cost) {
                            for(int k=0;k<now_num;k++){    //change back
                                cluster_labels[grid_asses[ori_gid1[k]]] = lb1;
                            }
                        }else {
                            label_change_cnt[mainLabel1] -= now_num;
                            label_change_cnt[mainLabel2] += now_num;
                            update_flag = 1;
                            improve_flag = 1;
                        }
                    }
                }
            }

            delete[] label_pairs;
            delete[] order1;
            delete[] labels_cnt;

            if(improve_flag==1){
                downCnt=0;
            }else downCnt += 1;

            if(downCnt>=downMax) {
//                if(type=="PerimeterRatio")continue;
                break;
            }
        }
    }

    // ----------------------------------swap step done----------------------------------------

    std::vector<int> ret(num, 0);
    for(int i=0;i<N;i++)ret[i] = cluster_labels[i];

    delete[] checked;

    delete[] old_grid_asses;

    delete[] grid_asses;
    delete[] cluster_labels;
    delete[] change;
    delete[] change_grid;
    delete[] consider;
    delete[] consider_e;
    delete[] consider_grid;

    delete[] Similar_cost_matrix;
    delete[] Compact_cost_matrix;
    delete[] ans;

    delete[] label_change_cnt;

    return ret;
}


// check cost
std::vector<double> Optimizer::checkCostForOne(
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const std::string &type,
double alpha, double beta,
int min_grids=0) {
    min_grids = std::max(1, min_grids);

    int N = _grid_asses.size();
    int num = _cluster_labels.size();
    int square_len = ceil(sqrt(N));
    int maxLabel = 0;
    for(int i=0;i<num;i++)maxLabel = std::max(maxLabel, _cluster_labels[i]+1);
    int *grid_asses = new int[N];
//    int *ori_grid_asses = new int[N];
    double (*ori_embedded)[2] = new double[N][2];
    int *cluster_labels = new int[num];
    for(int i=0;i<N;i++)grid_asses[i] = _grid_asses[i];
//    for(int i=0;i<N;i++)ori_grid_asses[i] = _ori_grid_asses[i];
    for(int i=0;i<N;i++) {
        ori_embedded[i][0] = _ori_embedded[i][0];
        ori_embedded[i][1] = _ori_embedded[i][1];
    }
    for(int i=0;i<num;i++)cluster_labels[i] = _cluster_labels[i];

    double *Similar_cost_matrix = new double[N*N];
    getOriginCostMatrixArrayToArray(ori_embedded, cluster_labels, Similar_cost_matrix, N, num, square_len, maxLabel);

    double *Compact_cost_matrix = new double[N*N];
    getCompactCostMatrixArrayToArray(grid_asses, cluster_labels, Compact_cost_matrix, N, num, square_len, maxLabel);

    double *old_T_pair = new double[2];    // convexity of triples
    double *old_D_pair = new double[N*N*2];
    int *old_grid_asses = new int[N];
    for(int i=0;i<N;i++)old_grid_asses[i] = grid_asses[i];

    bool *if_disconn = new bool[N];   // disconnect of now grids
    for(int i=0;i<N;i++)if_disconn[i] = false;
    int *checked = new int[N];
    double c_cost = checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, if_disconn);

    std::vector<double> ret(0, 0);
    for(int lb=0;lb<maxLabel;lb++) {
//        printf("start lb %d\n", lb);
        int *ans = new int[N];
        for(int gid=0;gid<N;gid++)ans[gid] = -1;
        int cnt = 0;
        for(int gid=0;gid<N;gid++)
        if((if_disconn[gid]==false)&&((grid_asses[gid]<num)&&(cluster_labels[grid_asses[gid]]==lb))) {
            ans[gid] = cnt;
            cnt += 1;
        }
        if(cnt<min_grids) {
            delete[] ans;
            continue;
        }

        int cnt2 = cnt;
        int *ans_labels = new int[cnt];

        for(int id=0;id<cnt;id++)ans_labels[id] = 0;
        for(int gid=0;gid<N;gid++)
        if(ans[gid]==-1) {
            ans[gid] = cnt2;
            cnt2 += 1;
        }

        std::vector<double> cost(4, -1);
        if(type=="CutRatio")
            cost = checkCostForE(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta);
        else if(type=="AreaRatio")
            cost = checkCostForS(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta);
        else if(type=="PerimeterRatio")
            cost = checkCostForC(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta);
        else if(type=="2020")
            cost = checkCostFor2020(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta);
        else if(type=="TB")
            cost = checkCostForTB(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta);
        else if(type=="Triple") {
            cost = checkCostForT(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta, true, false, old_grid_asses, old_T_pair);
            for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = ans[tmp_gid];
        }
        else if(type=="T2") {
            cost = checkCostForT2(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta, true, false, old_grid_asses, old_T_pair, old_D_pair);
            for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = ans[tmp_gid];
        }
        else if(type=="B")
            cost = checkCostForB(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta);
        else if(type=="Dev")
            cost = checkCostForDev(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta);
        else if(type=="AlphaT")
            cost = checkCostForAlphaT(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta);
        else if(type=="MoS")
            cost = checkCostForMoS(Similar_cost_matrix, Compact_cost_matrix, ans, ans_labels, N, cnt, square_len, 1, alpha, beta);
        // printf("cost %.6lf %.6lf %.6lf\n", cost[1], cost[2], cost[3]);


        ret.push_back(cost[3]);

        delete[] ans;
        delete[] ans_labels;
    }

    delete[] if_disconn;
    delete[] checked;

    delete[] old_grid_asses;
    delete[] old_T_pair;
    delete[] old_D_pair;

    delete[] grid_asses;
    delete[] ori_embedded;
    delete[] cluster_labels;
    delete[] Similar_cost_matrix;
    delete[] Compact_cost_matrix;

    return ret;
}

// check cost
std::vector<double> Optimizer::checkCostForAllShapes(
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const std::string &type,
double alpha, double beta,
int min_grids=0) {
    min_grids = std::max(1, min_grids);

    int N = _grid_asses.size();
    int num = _cluster_labels.size();
    int square_len = ceil(sqrt(N));
    int maxLabel = 0;
    for(int i=0;i<num;i++)maxLabel = std::max(maxLabel, _cluster_labels[i]+1);
    int *grid_asses = new int[N];
//    int *ori_grid_asses = new int[N];
    double (*ori_embedded)[2] = new double[N][2];
    int *cluster_labels = new int[num];
    for(int i=0;i<N;i++)grid_asses[i] = _grid_asses[i];
//    for(int i=0;i<N;i++)ori_grid_asses[i] = _ori_grid_asses[i];
    for(int i=0;i<N;i++) {
        ori_embedded[i][0] = _ori_embedded[i][0];
        ori_embedded[i][1] = _ori_embedded[i][1];
    }
    for(int i=0;i<num;i++)cluster_labels[i] = _cluster_labels[i];

    double *Similar_cost_matrix = new double[N*N];
    getOriginCostMatrixArrayToArray(ori_embedded, cluster_labels, Similar_cost_matrix, N, num, square_len, maxLabel);

    double *Compact_cost_matrix = new double[N*N];
    getCompactCostMatrixArrayToArray(grid_asses, cluster_labels, Compact_cost_matrix, N, num, square_len, maxLabel);

    double *old_T_pair = new double[2];    // convexity of triples
    double *old_D_pair = new double[N*N*2];
    int *old_grid_asses = new int[N];
    for(int i=0;i<N;i++)old_grid_asses[i] = grid_asses[i];

    bool *if_disconn = new bool[N];   // disconnect of now grids
    for(int i=0;i<N;i++)if_disconn[i] = false;
    int *checked = new int[N];
    double c_cost = checkConnectForAll(grid_asses, cluster_labels, checked, N, num, square_len, maxLabel, 8, if_disconn);

    std::vector<double> ret(0, 0);
    int clusters_cnt = 0;
    int all_cnt = 0;
    int *all_ans = new int[N];
    int *all_labels = new int[N];
    for(int gid=0;gid<N;gid++)all_ans[gid] = -1;

    for(int lb=0;lb<maxLabel;lb++) {
//        printf("start lb %d\n", lb);
        int cnt = 0;
        for(int gid=0;gid<N;gid++)
        if((if_disconn[gid]==false)&&((grid_asses[gid]<num)&&(cluster_labels[grid_asses[gid]]==lb))) {
            cnt += 1;
        }
        if(cnt<min_grids) {
            continue;
        }

        for(int gid=0;gid<N;gid++)
        if((if_disconn[gid]==false)&&((grid_asses[gid]<num)&&(cluster_labels[grid_asses[gid]]==lb))) {
            all_ans[gid] = all_cnt;
            all_labels[all_cnt] = clusters_cnt;
            all_cnt += 1;
        }
        clusters_cnt += 1;
    }

    int all_cnt2 = all_cnt;

    for(int gid=0;gid<N;gid++)
    if(all_ans[gid]==-1) {
        all_ans[gid] = all_cnt2;
        all_cnt2 += 1;
    }

    if(all_cnt==0) {
        delete[] all_ans;
        delete[] all_labels;

        delete[] if_disconn;
        delete[] checked;

        delete[] old_grid_asses;
        delete[] old_T_pair;
        delete[] old_D_pair;

        delete[] grid_asses;
        delete[] ori_embedded;
        delete[] cluster_labels;
        delete[] Similar_cost_matrix;
        delete[] Compact_cost_matrix;
        return ret;
    }

    std::vector<double> cost(4, -1);

    if(type=="CutRatio")
        cost = checkCostForE(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta);
    else if(type=="AreaRatio")
        cost = checkCostForS(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta);
    else if(type=="PerimeterRatio")
        cost = checkCostForC(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta);
    else if(type=="2020")
        cost = checkCostFor2020(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta);
    else if(type=="TB")
        cost = checkCostForTB(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta);
    else if(type=="Triple") {
        cost = checkCostForT(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta, true, false, old_grid_asses, old_T_pair);
        for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = all_ans[tmp_gid];
    }
    else if(type=="T2") {
        cost = checkCostForT2(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta, true, false, old_grid_asses, old_T_pair, old_D_pair);
        for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = all_ans[tmp_gid];
    }
    else if(type=="B")
        cost = checkCostForB(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta);
    else if(type=="Dev")
        cost = checkCostForDev(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta);
    else if(type=="AlphaT")
        cost = checkCostForAlphaT(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta);
    else if(type=="MoS")
        cost = checkCostForMoS(Similar_cost_matrix, Compact_cost_matrix, all_ans, all_labels, N, all_cnt, square_len, clusters_cnt, alpha, beta);
        // printf("cost %.6lf %.6lf %.6lf\n", cost[1], cost[2], cost[3]);

    ret.push_back(cost[3]);

    delete[] all_ans;
    delete[] all_labels;

    delete[] if_disconn;
    delete[] checked;

    delete[] old_grid_asses;
    delete[] old_T_pair;
    delete[] old_D_pair;

    delete[] grid_asses;
    delete[] ori_embedded;
    delete[] cluster_labels;
    delete[] Similar_cost_matrix;
    delete[] Compact_cost_matrix;

    return ret;
}

// check convexity one cluster
std::vector<double> Optimizer::checkCostForAll(
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const std::string &type,
double alpha, double beta,
const std::vector<bool> &_consider) {

    // ----------------------------------preprocess step start----------------------------------------

    int N = _grid_asses.size();
    int num = _cluster_labels.size();
    int square_len = ceil(sqrt(N));
    int maxLabel = 0;
    for(int i=0;i<num;i++)maxLabel = std::max(maxLabel, _cluster_labels[i]+1);
    int *grid_asses = new int[N];
//    int *ori_grid_asses = new int[N];
    double (*ori_embedded)[2] = new double[N][2];
    int *cluster_labels = new int[num];
    for(int i=0;i<N;i++)grid_asses[i] = _grid_asses[i];
//    for(int i=0;i<N;i++)ori_grid_asses[i] = _ori_grid_asses[i];
    for(int i=0;i<N;i++) {
        ori_embedded[i][0] = _ori_embedded[i][0];
        ori_embedded[i][1] = _ori_embedded[i][1];
    }
    for(int i=0;i<num;i++)cluster_labels[i] = _cluster_labels[i];
    int *consider = new int[N];
    int *consider_e = new int[N];
    int consider_N = 0;
    for(int i=0;i<N;i++){
        consider[i] = (_consider[i]?1:0);
        consider_e[grid_asses[i]] = consider[i];
        consider_N += consider[i];
    }

    double *Similar_cost_matrix = new double[N*N];
    getOriginCostMatrixArrayToArray(ori_embedded, cluster_labels, Similar_cost_matrix, N, num, square_len, maxLabel, consider);

    double *Compact_cost_matrix = new double[N*N];
    getCompactCostMatrixArrayToArray(grid_asses, cluster_labels, Compact_cost_matrix, N, num, square_len, maxLabel, consider);

    int *ans = new int[N];
    for(int i=0;i<N;i++)ans[i] = grid_asses[i];
    double best = 2147483647;    // cost of ans

    double *old_T_pair = new double[2];    // convexity of triples
    double *old_D_pair = new double[N*N*2];
    int *old_grid_asses = new int[N];
    for(int i=0;i<N;i++)old_grid_asses[i] = grid_asses[i];

    if(show_info)
        printf("final cost\n");

    std::vector<double> cost(4, -1);
    if(type=="CutRatio")
        cost = checkCostForE(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);
    else if(type=="AreaRatio")
        cost = checkCostForS(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider);
    else if(type=="PerimeterRatio")
        cost = checkCostForC(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider);
    else if(type=="2020")
        cost = checkCostFor2020(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider);
    else if(type=="TB")
        cost = checkCostForTB(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, false, false, nullptr, -1, -1, consider);
    else if(type=="Triple") {
        cost = checkCostForT(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, false, old_grid_asses, old_T_pair, consider);
        for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = ans[tmp_gid];
    }
    else if(type=="T2") {
        cost = checkCostForT2(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, true, false, old_grid_asses, old_T_pair, old_D_pair, consider);
        for(int tmp_gid=0;tmp_gid<N;tmp_gid++)old_grid_asses[tmp_gid] = ans[tmp_gid];
    }
    else if(type=="B")
        cost = checkCostForB(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);
    else if(type=="Dev")
        cost = checkCostForDev(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);
    else if(type=="Global")
        cost = checkCostForGlobal(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);
    else if(type=="AlphaT")
        cost = checkCostForAlphaT(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);
    else if(type=="MoS")
        cost = checkCostForMoS(Similar_cost_matrix, Compact_cost_matrix, ans, cluster_labels, N, num, square_len, maxLabel, alpha, beta, consider);

    std::vector<double> ret(N+4, 0);
    for(int i=0;i<N;i++)ret[i] = ans[i];
    for(int i=0;i<3;i++)ret[N+i] = cost[i+1];
    int* checked = new int[N];
    double tmp = N*checkConnectForAll(ans, cluster_labels, checked, N, num, square_len, maxLabel, 8, nullptr, consider_e);
    ret[N+3] = tmp;
    delete[] checked;

    delete[] old_grid_asses;
    delete[] old_T_pair;
    delete[] old_D_pair;

    delete[] grid_asses;
//    delete[] ori_grid_asses;
    delete[] ori_embedded;
    delete[] cluster_labels;
    delete[] Similar_cost_matrix;
    delete[] Compact_cost_matrix;
    delete[] ans;

    delete[] consider;
    delete[] consider_e;

    return ret;
}

// bi-graph match in each cluster to ensure the minist Similar cost
std::vector<int> Optimizer::optimizeInnerCluster(
//const std::vector<int> &_ori_grid_asses,
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const std::vector<bool> &_change, 
const std::vector<bool> &_consider) {
    double start = clock();
    int N = _grid_asses.size();
    int num = _cluster_labels.size();
    int square_len = ceil(sqrt(N));
    int maxLabel = 0;
    for(int i=0;i<num;i++)maxLabel = std::max(maxLabel, _cluster_labels[i]+1);
    int *grid_asses = new int[N];
//    int *ori_grid_asses = new int[N];
    double (*ori_embedded)[2] = new double[N][2];
    int *cluster_labels = new int[num];
    bool *change = new bool[N];
    int *consider = new int[N];
    int *consider_e = new int[N];
    for(int i=0;i<N;i++)grid_asses[i] = _grid_asses[i];
//    for(int i=0;i<N;i++)ori_grid_asses[i] = _ori_grid_asses[i];
    for(int i=0;i<N;i++) {
        ori_embedded[i][0] = _ori_embedded[i][0];
        ori_embedded[i][1] = _ori_embedded[i][1];
    }
    for(int i=0;i<num;i++)cluster_labels[i] = _cluster_labels[i];
    for(int i=0;i<N;i++)change[i] = _change[i];
    int consider_N = 0;
    for(int i=0;i<N;i++){
        consider[i] = (_consider[i]?1:0);
        consider_e[grid_asses[i]] = consider[i];
        consider_N += consider[i];
    }

    double *Similar_cost_matrix = new double[N*N];
    getOriginCostMatrixArrayToArray(ori_embedded, cluster_labels, Similar_cost_matrix, N, num, square_len, maxLabel, consider);

    double *cost_matrix = new double[N*N];    //cluster内部各自进行二分图匹配，代价矩阵
    #pragma omp parallel for num_threads(THREADS_NUM)
    for(int gid2=0;gid2<N;gid2++){
        int lb2 = -1;
        int id2 = grid_asses[gid2];
        if(id2<num)lb2 = cluster_labels[id2];
        for(int gid1=0;gid1<N;gid1++){
            int lb1 = -1;
            int id1 = grid_asses[gid1];
            if(id1<num)lb1 = cluster_labels[id1];
            if(lb1==lb2){
                cost_matrix[gid2*N+id1] = Similar_cost_matrix[gid2*N+id1];
            }else {
                cost_matrix[gid2*N+id1] = N;
            }
        }
    }

    std::vector<int> ret = solveBiMatchChange(cost_matrix, N, change, grid_asses);   //cluster内部各自进行二分图匹配

    delete[] cost_matrix;
    delete[] grid_asses;
//    delete[] ori_grid_asses;
    delete[] ori_embedded;
    delete[] cluster_labels;
    delete[] change;
    delete[] consider;
    delete[] consider_e;
    delete[] Similar_cost_matrix;

    return ret;
}

// bi-graph match in each cluster to ensure the minist Similar cost, with must-link
std::vector<int> Optimizer::optimizeInnerClusterWithMustLink(
//const std::vector<int> &_ori_grid_asses,
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const std::vector<bool> &_change,
const std::vector<std::vector<int>> &_must_links,
const std::vector<std::vector<int>> &_must_links2,
const int maxit = 1) {
    int N = _grid_asses.size();
    int num = _cluster_labels.size();
    int square_len = ceil(sqrt(N));
    int maxLabel = 0;
    for(int i=0;i<num;i++)maxLabel = std::max(maxLabel, _cluster_labels[i]+1);
    int *grid_asses = new int[N];
//    int *ori_grid_asses = new int[N];
    double (*ori_embedded)[2] = new double[N][2];
    int *cluster_labels = new int[num];
    bool *change = new bool[N];
    for(int i=0;i<N;i++)grid_asses[i] = _grid_asses[i];
//    for(int i=0;i<N;i++)ori_grid_asses[i] = _ori_grid_asses[i];
    for(int i=0;i<N;i++) {
        ori_embedded[i][0] = _ori_embedded[i][0];
        ori_embedded[i][1] = _ori_embedded[i][1];
    }
    for(int i=0;i<num;i++)cluster_labels[i] = _cluster_labels[i];
    for(int i=0;i<N;i++)change[i] = _change[i];

    double *Similar_cost_matrix = new double[N*N];
    getOriginCostMatrixArrayToArray(ori_embedded, cluster_labels, Similar_cost_matrix, N, num, square_len, maxLabel);

    double *cost_matrix = new double[N*N];    //cluster内部各自进行二分图匹配，代价矩阵
    int ml_N = _must_links.size();
    int ml_N2 = _must_links2.size();
    int *element_asses = new int[N];
    double *cluster_dist = new double[N*(maxLabel+1)];
    bool *is_border = new bool[N];
    for(int id=0;id<N;id++)is_border[id] = false;
    for(int i=0;i<ml_N2;i++)is_border[_must_links2[i][0]] = true;


    #pragma omp parallel for num_threads(THREADS_NUM)
    for(int gid2=0;gid2<N;gid2++){
        int bias = gid2*(maxLabel+1);
        int lb2 = maxLabel;
        int id2 = grid_asses[gid2];
        if(id2<num)lb2 = cluster_labels[id2];
        int x2 = gid2/square_len;
        int y2 = gid2%square_len;
        for(int lb=0;lb<=maxLabel;lb++)cluster_dist[bias+lb] = 2147483647;
        for(int gid1=0;gid1<N;gid1++){
            int lb1 = maxLabel;
            int id1 = grid_asses[gid1];
            if(id1<num)lb1 = cluster_labels[id1];
            int x1 = gid1/square_len;
            int y1 = gid1%square_len;
            cluster_dist[bias+lb1] = std::min(cluster_dist[bias+lb1], getDist(x1, y1, x2, y2));
        }
    }

    for(int it=0;it<maxit;it++) {
        #pragma omp parallel for num_threads(THREADS_NUM)
        for(int gid2=0;gid2<N;gid2++){
            int bias = gid2*N;
            int lb2 = maxLabel;
            int id2 = grid_asses[gid2];
            element_asses[id2] = gid2;
            if(id2<num)lb2 = cluster_labels[id2];
            for(int gid1=0;gid1<N;gid1++){
                int lb1 = maxLabel;
                int id1 = grid_asses[gid1];
                if(id1<num)lb1 = cluster_labels[id1];
                if(lb1==lb2){
                    cost_matrix[bias+id1] = Similar_cost_matrix[bias+id1];
                }else {
                    cost_matrix[bias+id1] = N;
                }
            }
        }

        for(int i=0;i<ml_N;i++) {
            int ml_size = _must_links[i].size();
            double x = 0;
            double y = 0;
            double tot_weight = 0;
            for(int j=0;j<ml_size;j++) {
                int id = _must_links[i][j];
                double weight = 1;
                if(is_border[id])weight = 2*(ml_size-1);
                int gid = element_asses[id];
                x += weight*(gid/square_len);
                y += weight*(gid%square_len);
                tot_weight += weight;
            }
            x /= tot_weight;
            y /= tot_weight;
            #pragma omp parallel for num_threads(THREADS_NUM)
            for(int j=0;j<ml_size;j++) {
                int id = _must_links[i][j];
                for(int gid=0;gid<N;gid++) {
                    int x0 = gid/square_len;
                    int y0 = gid%square_len;
                    cost_matrix[gid*N+id] += 1 * ((x0-x)*(x0-x) + (y0-y)*(y0-y));
                }
            }
        }

        for(int i=0;i<ml_N2;i++) {
            int id = _must_links2[i][0];
            int lb = _must_links2[i][1];
            for(int gid=0;gid<N;gid++){
                int tmp = gid*(maxLabel+1)+lb;
                cost_matrix[gid*N+id] += 10*cluster_dist[tmp]*cluster_dist[tmp];
            }
        }

        std::vector<int> ret = solveBiMatchChange(cost_matrix, N, change, grid_asses);   //cluster内部各自进行二分图匹配
        for(int i=0;i<N;i++)grid_asses[i] = ret[i];
    }
    std::vector<int> ret(N, 0);
    for(int i=0;i<N;i++)ret[i] = grid_asses[i];

    delete[] cost_matrix;
    delete[] grid_asses;
//    delete[] ori_grid_asses;
    delete[] ori_embedded;
    delete[] cluster_labels;
    delete[] change;
    delete[] Similar_cost_matrix;
    delete[] element_asses;
    delete[] cluster_dist;
    delete[] is_border;

    return ret;
}

void testmem() {
    double a[4] = {1, 2, 3, 4};
    memset(a, 0, 2*sizeof(double));
    for(int i=0;i<4;i++)printf("%.2lf\n", a[i]);
}

struct testo {
    testo(const int& _pi){ pi = _pi; }
    void testprintf() {
        for(int i=0;i<10;i++){
            printf("%d %d\n", pi, i);
//            Sleep(1000);
        }
    }
    int pi;
};

std::vector<double> checkNeighbor(std::vector<std::vector<int>> a, std::vector<std::vector<int>> b, int k, std::vector<bool> check, std::string kind="all", bool same_kind=false) {
    std::vector<double> ret(k);
    int* check_a = new int[a.size()];
    int* check_b = new int[a.size()];
    for(int i=0;i<a.size();i++){
        std::remove(std::begin(a[i]), std::end(a[i]), i);
        std::remove(std::begin(b[i]), std::end(b[i]), i);
    }
    double *all_tot = new double[k];
    int *all_full_cnt = new int[k];
    for(int ki=0;ki<k;ki++) {
        all_tot[ki] = all_full_cnt[ki] = 0;
    }
    for(int j=0;j<a.size();j++){
        check_a[j] = -1;
        check_b[j] = -1;
    }
    for(int i=0;i<a.size();i++){
        if(!check[i])continue;
        double tot=0;
        int full_cnt = 0;
        full_cnt += 1;
        int j1 = 0, cnt1 = 0;
        int j2 = 0, cnt2 = 0;
        for(int ki=0;ki<k;ki++) {
            // full_cnt += 1;
            while((cnt1<ki+1)&&(j1<a.size())){
                if((kind=="all")||((kind=="same")&&(check[a[i][j1]]))||((kind=="cross")&&(!check[a[i][j1]]))) {
                    check_a[a[i][j1]] = i;
                    cnt1 += 1;
                    if((check_a[a[i][j1]]==i)&&(check_b[a[i][j1]]==i))tot += 1;
                } else {
                    if(!same_kind) cnt1 += 1;
                }
                j1 += 1;
            }
            while((cnt2<ki+1)&&(j2<a.size())){
                if((kind=="all")||((kind=="same")&&(check[b[i][j2]]))||((kind=="cross")&&(!check[b[i][j2]]))) {
                    check_b[b[i][j2]] = i;
                    cnt2 += 1;
                    if((check_a[b[i][j2]]==i)&&(check_b[b[i][j2]]==i))tot += 1;
                } else {
                    if(!same_kind) cnt2 += 1;
                }
                j2 += 1;
            }
            // for(int j=0;j<a.size();j++){
            //     if((check_a[j]==1)&&(check_b[j]==1)) tot += 1;
            // }
            all_tot[ki] += tot;
            all_full_cnt[ki] += full_cnt;
        }
    }
    for(int ki=0;ki<k;ki++) {
        ret[ki] = all_tot[ki]/all_full_cnt[ki];
    }
    delete[] check_a;
    delete[] check_b;
    delete[] all_tot;
    delete[] all_full_cnt;
    return ret;
}

std::vector<double> checkENeighbor(std::vector<std::vector<int>> a, std::vector<std::vector<double>> dist_a, std::vector<std::vector<int>> b, double max_e, int k, std::vector<bool> check, std::string kind="all", bool same_kind=false) {
    std::vector<double> ret(k*2);
    int* check_a = new int[a.size()];
    int* check_b = new int[a.size()];
    for(int i=0;i<a.size();i++){
        std::remove(std::begin(a[i]), std::end(a[i]), i);
        std::remove(std::begin(b[i]), std::end(b[i]), i);
    }
    for(int ki=0;ki<k;ki++) {
        double e=max_e/k*(ki+1);
        double tot=0;
        double full_cnt = 0;
        for(int i=0;i<a.size();i++){
            if(!check[i])continue;
            for(int j=0;j<a.size();j++){
                check_a[j] = 0;
                check_b[j] = 0;
            }
            int j = 0, cnt = 0;
            while((j<a[i].size())&&(dist_a[i][a[i][j]]<=e)){
//            while((j<a[i].size())&&(cnt<=ki)){
                if((kind=="all")||((kind=="same")&&(check[a[i][j]]))||((kind=="cross")&&(!check[a[i][j]]))) {
                    check_a[a[i][j]] = 1;
                    cnt += 1;
                } else {
                    if(!same_kind) cnt += 1;
                }
                j += 1;
            }
            j = 0; int cnt2 = 0;
            while(cnt2<cnt){
                if((kind=="all")||((kind=="same")&&(check[b[i][j]]))||((kind=="cross")&&(!check[b[i][j]]))) {
                    check_b[b[i][j]] = 1;
                    cnt2 += 1;
                } else {
                    if(!same_kind) cnt2 += 1;
                }
                j += 1;
            }
            for(int j=0;j<a.size();j++){
                if((check_a[j]==1)&&(check_b[j]==1)) tot += 1;
            }
            full_cnt += cnt;
        }
        if(full_cnt>0)ret[ki*2] = tot/full_cnt;
        else ret[ki*2] = 0;
        ret[ki*2+1] = full_cnt;
    }
    delete[] check_a;
    delete[] check_b;
    return ret;
}

PYBIND11_MODULE(gridlayoutOpt, m) {
    m.doc() = "Gridlayout Optimizer"; // optional module docstring
    py::class_<Optimizer>(m, "Optimizer")
    	.def(py::init<const int&>())
        .def("getClusters", &Optimizer::getClusters, "A function")
        .def("solveKM", &Optimizer::solveKM, "A function")
        .def("solveKMLabel", &Optimizer::solveKMLabel, "A function")
        .def("solveLap", &Optimizer::solveLap, "A function")
        .def("optimizeBA", &Optimizer::optimizeBA, "A function to optimize")
        .def("optimizeSwap", &Optimizer::optimizeSwap, "A function to optimize")
        .def("checkCostForAll", &Optimizer::checkCostForAll, "A function to check cost")
        .def("checkCostForOne", &Optimizer::checkCostForOne, "A function to check cost")
        .def("checkCostForAllShapes", &Optimizer::checkCostForAllShapes, "A function to check cost")
        .def("optimizeInnerCluster", &Optimizer::optimizeInnerCluster, "A function to optimize")
        .def("optimizeInnerClusterWithMustLink", &Optimizer::optimizeInnerClusterWithMustLink, "A function to optimize");
    m.def("grid_op", &grid_op, "A function to optimize");
    m.def("grid_op_partition", &grid_op_partition, "A function to optimize");
    m.def("grid_assign_partition", &grid_assign_partition, "A function to optimize");
    m.def("testmem", &testmem, "A function to test");
//    m.def("testprintf", &testprintf, "A function to test");
    py::class_<testo>(m, "testo")
    	.def(py::init<const int&>())
		.def("testprintf", &testo::testprintf);
    m.def("changeLabelsForConvexity", &changeLabelsForConvexity, "A function to test");
    m.def("checkNeighbor", &checkNeighbor, "A function to test");
    m.def("checkENeighbor", &checkENeighbor, "A function to test");
    m.def("getConnectBorder", &getConnectBorder, "A function to get border elements");
    m.def("getConnectShape", &getConnectShape, "A function to get border elements");
    m.def("getMainConnectGrids", &getMainConnectGrids, "A function to get border elements");
    m.def("SearchForTree", &SearchForTree, "A function to get treemap");
    m.def("getConvexityForTriplesFree", &getConvexityForTriplesFree, "A function");
    m.def("getConvexityForPerimeterFree", &getConvexityForPerimeterFree, "A function");
}