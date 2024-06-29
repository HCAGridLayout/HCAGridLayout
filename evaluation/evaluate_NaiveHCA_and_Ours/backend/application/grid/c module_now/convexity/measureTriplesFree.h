#ifndef _MEASURE_TRIPLES_FREE_H
#define _MEASURE_TRIPLES_FREE_H

#include "../utils/base.h"
#include "../utils/convexHull.h"
#include "../utils/util.h"

bool inRect(double x, double y, double rect[][2]) {
    int j = 3;
    bool oddNodes = false;
    for(int i=0;i<4;i++) {
        double xi = rect[i][0];
        double yi = rect[i][1];
        double xj = rect[j][0];
        double yj = rect[j][1];

        if ((yi-0.000001<y&&yj+0.000001>y)||(yj-0.000001<y&&yi+0.000001>y)) {
            if(abs(yi-yj)<0.000001){
                if((std::min(xi, xj)-0.000001<x)&&(x<std::max(xi, xj)+0.000001))return false;
            }else if(abs(xi+(y-yi)/(yj-yi)*(xj-xi)-x)<0.000001) return false;
        }

        if ((yi<y&&yj>=y)||(yj<y&&yi>=y)) {
            if(xi+(y-yi)/(yj-yi)*(xj-xi)<x) {
                oddNodes = !oddNodes;
            }
        }
        j = i;
    }
    return oddNodes;
}

double getConvexityForTriplesFree(
const std::vector<std::vector<int>> &_points,
const std::vector<int> &_cluster_labels,
const int grid_size) {
    int N = _points.size();
    int (*points)[2] = new int[N][2];
    int* cluster_labels = new int[N];
    int* order = new int[N];
    for(int i=0;i<N;i++) {
        points[i][0] = _points[i][0];
        points[i][1] = _points[i][1];
        cluster_labels[i] = _cluster_labels[i];
        order[i] = i;
    }
    for(int i=0;i<N-1;i++)
    for(int j=i+1;j<N;j++)
    if(points[order[i]][0]>points[order[j]][0])std::swap(order[i], order[j]);

    double fcnt = 0;
    double wcnt = 0;
    int range_left=0;
    for(int i=0;i<N-1;i++){
        int id1 = order[i];
        while(points[order[range_left]][0]<=points[id1][0]-grid_size*0.5)range_left += 1;
        for(int j=i+1;j<N;j++){
            int id2 = order[j];
            if(cluster_labels[id1]!=cluster_labels[id2])continue;
            double rect[4][2];
            if(points[id1][0]>points[id2][0]){
                rect[0][0] = points[id1][0]+grid_size*0.5; rect[0][1] = points[id1][1]+grid_size*0.5;
                rect[1][0] = points[id1][0]-grid_size*0.5; rect[1][1] = points[id1][1]-grid_size*0.5;
                rect[2][0] = points[id2][0]-grid_size*0.5; rect[2][1] = points[id2][1]-grid_size*0.5;
                rect[3][0] = points[id2][0]+grid_size*0.5; rect[3][1] = points[id2][1]+grid_size*0.5;
            }else {
                rect[0][0] = points[id1][0]-grid_size*0.5; rect[0][1] = points[id1][1]+grid_size*0.5;
                rect[1][0] = points[id1][0]+grid_size*0.5; rect[1][1] = points[id1][1]-grid_size*0.5;
                rect[2][0] = points[id2][0]+grid_size*0.5; rect[2][1] = points[id2][1]-grid_size*0.5;
                rect[3][0] = points[id2][0]-grid_size*0.5; rect[3][1] = points[id2][1]+grid_size*0.5;
            }
            for(int k=range_left;(k<N)&&(points[order[k]][0]<points[id2][0]+grid_size*0.5);k++){
//                printf("check %d %d %d\n", i, j, k);
                if((k==i)||(k==j))continue;
                int id0 = order[k];
                if(inRect(points[id0][0], points[id0][1], rect)) {
                    fcnt += 1;
                    if(cluster_labels[id0]!=cluster_labels[id1])wcnt += 1;
                }
            }
        }
    }

    delete[] points;
    delete[] cluster_labels;
    delete[] order;
    return wcnt/fcnt;
}

#endif