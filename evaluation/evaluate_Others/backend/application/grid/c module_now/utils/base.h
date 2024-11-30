#ifndef _BASE_H
#define _BASE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <utility>
#include <ctime>

#define THREADS_NUM 48

#define M_PI 3.1415926535898
#define M_PI_2 3.1415926535898/2

#include "geometry.h"

namespace py = pybind11;
using std::cout;

enum Direction {
    GO_UP,
    GO_DOWN,
    GO_LEFT,
    GO_RIGHT
};

inline double sqr(double x) {
    return x * x;
}

double getDist(double x1, double y1, double x2, double y2){
    return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

int getLabel(
const std::vector<std::vector<int>> &grid_label,
const int &x, const int &y) {
    int n = grid_label.size();
    if (x < 0 || y < 0 || x >= n || y >= n) return -1;
    return grid_label[x][y];
}

struct Optimizer {
    Optimizer(const int& _id) { id = _id;}
    int id;

    ~Optimizer() {
        if(global_cnt>=0) {
            delete[] global_triples;
            delete[] global_triples_head;
            delete[] global_triples_list;
        }
    }

    // measureCHC.h

    std::vector<double> checkConvexForCArray(
        const int grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        bool save, bool load, double label_pairs[][2], int mainLabel1, int mainLabel2,
        int consider[], int consider_N, int consider_grid[]);

    std::vector<double> checkCostForC(
        const double Similar_cost_matrix[],
        const double Compact_cost_matrix[],
        const int grid_asses[], const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        const double &alpha, const double &beta,
        bool save, bool load, double label_pairs[][2], int mainLabel1, int mainLabel2,
        int consider[], int consider_N, int consider_grid[]);


    // measureCHS.h

    std::vector<double> checkConvexForSArray(
        const int grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        bool save, bool load, double label_pairs[][2], int mainLabel1, int mainLabel2);

    std::vector<double> checkCostForS(
        const double Similar_cost_matrix[],
        const double Compact_cost_matrix[],
        const int grid_asses[], const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        const double &alpha, const double &beta,
        bool save, bool load, double label_pairs[][2], int mainLabel1, int mainLabel2,
        int consider[]);


    //measureEdge.h

    std::vector<double> checkConvexForESingle4(
    const int grid_asses[],
    const int cluster_labels[],
    const int x_pre[], const int y_pre[], const int &gid, const int &lb,
    const int &N, const int &num, const int &square_len, const int &maxLabel);

    std::vector<double> checkConvexForEArray(
    const int grid_asses[],
    const int cluster_labels[],
    const int &N, const int &num, const int &square_len, const int &maxLabel,
    bool save, int save_x_pre[], int save_y_pre[]);

    std::vector<double> checkConvexForE(
    const std::vector<int> &_grid_asses,
    const std::vector<int> &_cluster_labels);

    void getCostMatrixForEArrayToArray(
    int grid_asses[],
    int cluster_labels[],
    double cost_matrix_a[],
    const int &N, const int &num, const int &square_len, const int &maxLabel);

    std::vector<std::vector<double>> getCostMatrixForE(
    const std::vector<int> &_grid_asses,
    const std::vector<int> &_cluster_labels);

    std::vector<double> checkCostForE(
        const double Similar_cost_matrix[],
        const double Compact_cost_matrix[],
        const int grid_asses[], const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        const double &alpha, const double &beta,
        int consider[]);


    //measureTriples.h

    int (*global_triples)[4];   //saved triples
    long long *global_triples_head;   //heads of link list saving triples of every grid
    long long (*global_triples_list)[2];   //link list saving triples of every grid
    long long global_cnt = -1;    //count of triples
    int global_hash = -1;    //hash of consider
    int global_N = -1;    //grid size of triples saved now

    int getHash(int N, int consider[]);

    void checkTriplesOfLine2(
        int triples[][4],
        long long triples_head[],
        long long triples_list[][2],
        long long &cnt,
        const int &square_len,
        const int &gid1, const int &gid2,
        int oriX, int oriY, int targetX, int startX, int dx, int dy,
        int consider[]);

    void checkTriplesOfLine(
        int triples[][4],
        long long triples_head[],
        long long triples_list[][2],
        long long &cnt,
        const int &square_len,
        const int &gid1, const int &gid2,
        int consider[]);

    void getConvexForTOfLine2(
        double &T0,
        double &T1,
        const int grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len,
        const int &gid1, const int &gid2,
        int oriX, int oriY, int targetX, int startX, int dx, int dy,
        int consider[]);
    
    void getConvexForTOfLine(
        double &T0,
        double &T1,
        const int grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len,
        const int &gid1, const int &gid2,
        int consider[]);

    long long getTriples(
        int triples[][4],
        long long triples_head[],
        long long triples_list[][2],
        const int grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        int consider[]);
    
    std::vector<double> getTriplesAndConvexForT(
    const int grid_asses[],
    const int cluster_labels[],
    const int &N, const int &num, const int &square_len, const int &maxLabel,
    int consider[]);

    std::vector<int> checkTriplesByDict(
        int innerDict[], int outerDict[],
        const int grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        const int gid, const int lb);

    void updateTripleDict(
        int innerDict[], int outerDict[],
        const int triples[][4],
        const long long triples_head[],
        const long long triples_list[][2],
        const int grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        const int gid);

    void updateTripleDictwithOld(
        double innerDict[], double outerDict[],
        const int change_triples[][3],
        const double change_triples_times[],
        const long long change_triples_head[],
        const long long change_triples_list[][2],
        const int grid_asses[],
        const int old_grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        const int gid);

    std::vector<double> checkConvexForTArray(
        const int triples[][4],
        const long long &cnt,
        const int grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel);

    std::vector<double> checkConvexForT(
        const std::vector<int> &_grid_asses,
        const std::vector<int> &_cluster_labels);

    std::vector<double> checkConvexForTwithOld(
        int triples[][4],
        long long triples_head[],
        long long triples_list[][2],
        double old_T_pair[],
        const int grid_asses[],
        const int old_grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel);

    long long getChangeTriplesCnt(
        const long long cnt,
        int triples[][4],
        long long triples_head[],
        long long triples_list[][2],
        const int grid_asses[],
        const int old_grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel);

    long long getChangeTriples(
        int change_triples[][3],
        double change_triples_times[],
        long long change_triples_head[],
        long long change_triples_list[][2],
        const long long cnt,
        int triples[][4],
        long long triples_head[],
        long long triples_list[][2],
        const int grid_asses[],
        const int old_grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel);

    void getCostMatrixForTArrayToArray(
        int grid_asses[],
        int cluster_labels[],
        double cost_matrix_a[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        bool save, bool load, int old_innerDict[], int old_outerDict[],
        int old_grid_asses[], double now_T_pair[],
        int consider[]);

    std::vector<std::vector<double>> getCostMatrixForT(
        const std::vector<int> &_grid_asses,
        const std::vector<int> &_cluster_labels);

    std::vector<double> checkCostForT(
        const double Similar_cost_matrix[],
        const double Compact_cost_matrix[],
        const int grid_asses[], const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        const double &alpha, const double &beta,
        bool save, bool load, int old_grid_asses[], double old_T_pair[],
        int consider[]);


    // measureDoubles.h

    void getDoubles(double &T0, double &T1, double a, double b);

    int getDoubleId(int x, int y, int N);

    std::vector<double> checkConvexForT2Array(
        const int triples[][4],
        const long long &cnt,
        bool save,
        double save_D_pair[],  // N*N*2
        const int grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel);

    std::vector<double> checkConvexForT2withOld(
    int triples[][4],
    long long triples_head[],
    long long triples_list[][2],
    bool save,
    double old_T_pair[],
    double old_D_pair[],  // N*N*2
    const int grid_asses[],
    const int old_grid_asses[],
    const int cluster_labels[],
    const int &N, const int &num, const int &square_len, const int &maxLabel);

    std::vector<double> checkCostForT2(
    const double Similar_cost_matrix[],
    const double Compact_cost_matrix[],
    const int grid_asses[], const int cluster_labels[],
    const int &N, const int &num, const int &square_len, const int &maxLabel,
    const double &alpha, const double &beta,
    bool save, bool load, int old_grid_asses[], double old_T_pair[], double old_D_pair[],
    int consider[]);


    //newMeasureMoS.h

    std::vector<double> checkPolygonConvexByMoS(
        const PointList & polygon,
        const std::vector<PointList> & holes,
        double alpha);

    std::vector<double> checkConvexForMoS(
        const std::vector<int> &_grid_asses,
        const std::vector<int> &_cluster_labels,
        double alpha);

    std::vector<double> checkConvexForMoSArray(
        const int grid_asses[],
        const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel);

    std::vector<double> checkCostForMoS(
        const double Similar_cost_matrix[],
        const double Compact_cost_matrix[],
        const int grid_asses[], const int cluster_labels[],
        const int &N, const int &num, const int &square_len, const int &maxLabel,
        const double &alpha, const double &beta,
        int consider[]);


    // main.h

    bool show_info = false;

    std::vector<int> getClusters(
        const std::vector<int> &_grid_asses,
        const std::vector<int> &_labels);

    std::vector<int> solveLapArray(
        const double dist[],
        const int N,
        int k);

    std::vector<int> solveLap(
        const std::vector<std::vector<double>> &_dist,
        const bool use_knn, int k_value);

    void workback(int x,int lb[][3], int p2[]);

    std::vector<int> solveKMArray(
        const double dist[],
        const int n_row, const int n_col);

    std::vector<int> solveKMArrayLabel(
        const double dist[],
        const int n_row, const int n_col,
        const int maxLabel, const int labels[]);

    std::vector<int> solveKM(
        const std::vector<std::vector<double>> &_dist);

    std::vector<int> solveKMLabel(
        const std::vector<std::vector<double>> &_dist,
        const std::vector<int> &_labels);

    std::vector<int> solveBiMatchChange(
        const double dist[],
        const int N,
        const bool change[],
        const int grid_asses[],
        const std::string &type);

    std::vector<double> optimizeBA(
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
        const std::vector<int> &_other_label);

    std::vector<double> optimizeSwap(
        //const std::vector<int> &_ori_grid_asses,
        const std::vector<std::vector<double>> &_ori_embedded,
        const std::vector<int> &_grid_asses,
        const std::vector<int> &_cluster_labels,
        const std::vector<bool> &_change,
        const std::string &type,
        double alpha, double beta,
        int maxit, int choose_k, int seed, bool innerBiMatch, int swap_cnt,
        const std::vector<bool> &_consider,
        const std::vector<int> &_other_label);

    std::vector<double> checkCostForOne(
        const std::vector<std::vector<double>> &_ori_embedded,
        const std::vector<int> &_grid_asses,
        const std::vector<int> &_cluster_labels,
        const std::string &type,
        double alpha, double beta,
        int min_grids);

    std::vector<double> checkCostForAllShapes(
        const std::vector<std::vector<double>> &_ori_embedded,
        const std::vector<int> &_grid_asses,
        const std::vector<int> &_cluster_labels,
        const std::string &type,
        double alpha, double beta,
        int min_grids);

    std::vector<double> checkCostForAll(
        const std::vector<std::vector<double>> &_ori_embedded,
        const std::vector<int> &_grid_asses,
        const std::vector<int> &_cluster_labels,
        const std::string &type,
        double alpha, double beta,
        const std::vector<bool> &_consider);

    std::vector<int> optimizeInnerCluster(
        //const std::vector<int> &_ori_grid_asses,
        const std::vector<std::vector<double>> &_ori_embedded,
        const std::vector<int> &_grid_asses,
        const std::vector<int> &_cluster_labels,
        const std::vector<bool> &_change, 
        const std::vector<bool> &_consider);

    std::vector<int> optimizeInnerClusterWithMustLink(
        //const std::vector<int> &_ori_grid_asses,
        const std::vector<std::vector<double>> &_ori_embedded,
        const std::vector<int> &_grid_asses,
        const std::vector<int> &_cluster_labels,
        const std::vector<bool> &_change,
        const std::vector<std::vector<int>> &_must_links,
        const std::vector<std::vector<int>> &_must_links2,
        const int maxit);
};

#endif