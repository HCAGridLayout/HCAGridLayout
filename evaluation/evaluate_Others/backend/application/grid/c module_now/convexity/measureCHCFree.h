#ifndef _MEASURE_CHC_FREE_H
#define _MEASURE_CHC_FREE_H


#include "../utils/base.h"
#include "../utils/convexHull.h"
#include "../utils/util.h"

double getConvexityForPerimeterFree(
const std::vector<std::vector<int>> &_points,
const std::vector<int> &_cluster_labels,
const int grid_size) {
    int N = _points.size();
    int (*points)[2] = new int[N][2];
    int* cluster_labels = new int[N];
    for(int i=0;i<N;i++) {
        points[i][0] = _points[i][0];
        points[i][1] = _points[i][1];
        cluster_labels[i] = _cluster_labels[i];
    }
    int maxLabel = 0;
    for(int i=0;i<N;i++)maxLabel = std::max(maxLabel, cluster_labels[i]+1);

    double (*nodes)[2] = new double[N*4][2];
    double C0 = 0, C1 = 0;
    for(int li=0;li<maxLabel;li++){
        int cnt=0;
        int* idx = new int[N];
        int id_cnt = 0;

        double tmp_C0=0, tmp_C1=0;
        for(int i=0;i<N;i++) {
            if(cluster_labels[i]!=li)continue;
            idx[id_cnt] = i;
            id_cnt += 1;
            nodes[cnt][0] = points[i][0]; nodes[cnt][1] = points[i][1]; cnt++;
            nodes[cnt][0] = points[i][0]+grid_size; nodes[cnt][1] = points[i][1]; cnt++;
            nodes[cnt][0] = points[i][0]; nodes[cnt][1] = points[i][1]+grid_size; cnt++;
            nodes[cnt][0] = points[i][0]+grid_size; nodes[cnt][1] = points[i][1]+grid_size; cnt++;

            tmp_C1 += 4*grid_size;

            for(int j=0;j<N;j++) {
                if((i==j)||(cluster_labels[j]!=li))continue;
                if(std::abs(points[i][0]-points[j][0])+std::abs(points[i][1]-points[j][1])<1.5*grid_size){
                    tmp_C1 -= grid_size;
                }
            }
        }

        cnt = getConvexHull(cnt, nodes);

        tmp_C0 = getCofPoly(cnt, nodes);

        C0 += tmp_C0;
        C1 += tmp_C1;

        // printf("C %.2lf %.2lf\n", C0, C1);

        double* dist_mtx = new double[id_cnt*id_cnt];
        double* nearst_dist = new double[id_cnt];
        int* connected = new int[id_cnt];
        for(int ii=0;ii<id_cnt;ii++)
        for(int jj=0;jj<id_cnt;jj++){
            int i = idx[ii];
            int j = idx[jj];
            double d1 = std::max(0, std::abs(points[i][0]-points[j][0])-grid_size);
            double d2 = std::max(0, std::abs(points[i][1]-points[j][1])-grid_size);
            double dist = std::sqrt(d1*d1+d2*d2);
            dist_mtx[ii*id_cnt+jj] = dist;
        }
        connected[0] = 1; nearst_dist[0] = 0;
        for(int ii=1;ii<id_cnt;ii++){
            int i = idx[ii];
            connected[ii] = 0;
            nearst_dist[ii] = dist_mtx[ii];
        }
        for(int oo=1;oo<id_cnt;oo++){
            int new_ii = -1;
            for(int ii=0;ii<id_cnt;ii++){
                if(connected[ii]==1)continue;
                if((new_ii==-1)||(nearst_dist[new_ii]>nearst_dist[ii]))new_ii = ii;
            }
            connected[new_ii] = 1;
            C1 += 2*nearst_dist[new_ii];
            for(int ii=0;ii<id_cnt;ii++){
                if(connected[ii]==1)continue;
                nearst_dist[ii] = std::min(nearst_dist[ii], dist_mtx[new_ii*id_cnt+ii]);
            }
        }
        // printf("C %.2lf\n", C1);

        delete[] dist_mtx;
        delete[] nearst_dist;
        delete[] connected;
        delete[] idx;
    }

    std::vector<double> C_pair(2, 0);
    C_pair[0] = (C1-C0);
    C_pair[1] = C1;

    delete[] nodes;
    delete[] points;
    delete[] cluster_labels;
    return C_pair[0]/C_pair[1];
}

#endif