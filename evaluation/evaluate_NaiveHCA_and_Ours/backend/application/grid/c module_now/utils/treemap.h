#ifndef _TREEMAP_H
#define _TREEMAP_H

#include <iostream>
#include <vector>
#include <utility>
#include <ctime>
#include <algorithm>
#include <math.h>
#include <map>
#include "base.h"

#define SIZENUM 49

//double time0 = 0;
//double time1 = 0;
//double time2 = 0;
//double time3 = 0;
//double time4 = 0;
//double time5 = 0;

void AxisRank(int order[], int n, std::vector<int> &rank, int SetRank[]) {
    if(SetRank[0]!=-1) {
        memcpy(order, SetRank, n*sizeof(int));
        return;
    }
	for(int i=0;i<n-1;i++)
	for(int j=i+1;j<n;j++)
	if(rank[order[j]]<rank[order[i]]) {
		int tmp = order[i];
		order[i] = order[j];
		order[j] = tmp;
	}
    memcpy(SetRank, order, n*sizeof(int));
}


int getSizeId(double x, double y, std::vector<double> &size_list) {
	int id=0;
	int left = std::max(0, int(x/(x+y)*(SIZENUM+1))-1);
	int right = std::min(int(size_list.size()), left+2);
//	for(int i=1;i<size_list.size();i++)
	for(int i=left;i<right;i++)
	if(std::abs(size_list[i]-x/(x+y))<std::abs(size_list[id]-x/(x+y)))id = i;
	return id;
}


double SearchForTreeDP(std::vector<std::vector<double>> &xy, std::vector<std::vector<double>> &sxy,
std::vector<int> &rank_x, std::vector<int> &rank_y, std::vector<double> &weight,
long long now_set, int n, int size_id, std::vector<double> &size_list,
std::map<long long, int> &SetHash, int &SetCount,
int CutAxis[][SIZENUM], int CutPlace[][SIZENUM], double CutCost[][SIZENUM],
int leftSize[][SIZENUM], int rightSize[][SIZENUM],
int SetRankX[], int SetRankY[],
double real_x=0.5) {

//    double start = clock();

	if(SetHash.count(now_set)==0){
		SetHash[now_set] = SetCount;
		SetCount += 1;
	}
	int SetId = SetHash[now_set];

//	time0 += (clock()-start)/CLOCKS_PER_SEC;

//	start = clock();

	double size_x = size_list[size_id];
	double size_y = 1-size_x;

	int *order_x = new int[n];
	int *order_y = new int[n];
	int now_n = 0;
	for(int i=0;i<n;i++)
	if(now_set&(1ll<<i)) {
		order_x[now_n] = order_y[now_n] = i;
		now_n += 1;
	}

	double tot = 0;
	for(int i=0;i<now_n;i++)tot += weight[order_x[i]];

	if(now_n == 1) {
		CutAxis[SetId][size_id] = 0;
		CutPlace[SetId][size_id] = 0;
		size_x = real_x;
		size_y = 1-real_x;
		CutCost[SetId][size_id] = 0;
//		CutCost[SetId][size_id] = std::max(size_x/size_y, size_y/size_x);
//		CutCost[SetId][size_id] = std::max(size_x/size_y, size_y/size_x)*tot;
		delete[] order_x;
		delete[] order_y;
//		double cost = std::max(size_x/size_y, size_y/size_x);
		double cost = tot*tot*(size_x*size_x+size_y*size_y)/(size_x*size_y);
		return cost;
	}

	if(CutAxis[SetId][size_id]>=0) {
	    delete[] order_x;
	    delete[] order_y;
	    return CutCost[SetId][size_id];
	}

    AxisRank(order_x, now_n, rank_x, SetRankX+(SetId*n));
    AxisRank(order_y, now_n, rank_y, SetRankY+(SetId*n));

//	time1 += (clock()-start)/CLOCKS_PER_SEC;

//	start = clock();

    double *left = new double[now_n];
    double *right = new double[now_n];
    double *top = new double[now_n];
    double *bottom = new double[now_n];

	std::vector<double> cross;
	for(int axis=0;axis<2;axis++) {
	    int *order;
		if(axis==0)order = order_x;
		else order = order_y;

		double *small, *big;
		if(axis==0) {
		    small = left;
		    big = right;
		}else {
		    small = top;
		    big = bottom;
		}

		double tmp = -1;
		for(int i=0;i<now_n;i++){
			tmp = std::max(tmp, xy[order[i]][axis]+sxy[order[i]][axis]);
			small[i] = tmp;
		}
		tmp = 100000;
		for(int i=now_n-1;i>=0;i--){
			tmp = std::min(tmp, xy[order[i]][axis]-sxy[order[i]][axis]);
			big[i] = tmp;
		}
		for(int i=0;i<now_n-1;i++)cross.push_back(small[i]-big[i+1]);
	}

	std::sort(cross.begin(), cross.end());

//	time2 += (clock()-start)/CLOCKS_PER_SEC;

	long long now_set1, now_set2;
	for(int axis=0;axis<2;axis++) {

//	    double start = clock();

	    int *order;
		if(axis==0)order = order_x;
		else order = order_y;

		double *small, *big;
		if(axis==0) {
		    small = left;
		    big = right;
		}else {
		    small = top;
		    big = bottom;
		}

		now_set1 = now_set2 = 0;
		double tmp_count = 0;

//	    time3 += (clock()-start)/CLOCKS_PER_SEC;

		for(int i=0;i<now_n-1;i++) {

//	        double start = clock();

			now_set1 += (1ll)<<order[i];
			now_set2 = now_set-now_set1;
			tmp_count += weight[order[i]];
			double shel = cross[cross.size()/4];
//			shel = std::min(shel, (cross[0]+cross[cross.size()/2])/2);
			if(small[i]-big[i+1]<=shel+1e-3) {
				double new_size_x1 = size_x;
				double new_size_y1 = size_y;
				double new_size_x2 = size_x;
				double new_size_y2 = size_y;
				if(axis==0){
					new_size_x1 = size_x*tmp_count/tot;
					new_size_x2 = size_x*(1-tmp_count/tot);
				}else{
					new_size_y1 = size_y*tmp_count/tot;
					new_size_y2 = size_y*(1-tmp_count/tot);
				}
				int size_id1 = getSizeId(new_size_x1, new_size_y1, size_list);
				int size_id2 = getSizeId(new_size_x2, new_size_y2, size_list);

//	            time4 += (clock()-start)/CLOCKS_PER_SEC;

				double cost1 = SearchForTreeDP(xy, sxy, rank_x, rank_y, weight, now_set1, n, size_id1, size_list, SetHash, SetCount, CutAxis, CutPlace, CutCost, leftSize, rightSize, SetRankX, SetRankY, new_size_x1/(new_size_x1+new_size_y1));
				double cost2 = SearchForTreeDP(xy, sxy, rank_x, rank_y, weight, now_set2, n, size_id2, size_list, SetHash, SetCount, CutAxis, CutPlace, CutCost, leftSize, rightSize, SetRankX, SetRankY, new_size_x2/(new_size_x2+new_size_y2));

//	            start = clock();

//	            double cost = std::max(cost1, cost2);
				double cost = cost1 + cost2;
				if((CutAxis[SetId][size_id]<0)||(CutCost[SetId][size_id]>cost)) {
					CutAxis[SetId][size_id] = axis;
					CutPlace[SetId][size_id] = i;
					CutCost[SetId][size_id] = cost;
					leftSize[SetId][size_id] = size_id1;
					rightSize[SetId][size_id] = size_id2;
				}

//	            time5 += (clock()-start)/CLOCKS_PER_SEC;
			}
		}
	}

	delete[] order_x;
	delete[] order_y;

	delete[] left;
	delete[] right;
	delete[] top;
	delete[] bottom;

	return CutCost[SetId][size_id];
}


void getAns(std::vector<std::vector<double>> &xy, std::vector<std::vector<double>> &sxy,
std::vector<int> &rank_x, std::vector<int> &rank_y, std::vector<double> &weight,
long long now_set, int n, int size_id, std::vector<double> &size_list,
std::map<long long, int> &SetHash, int &SetCount,
int CutAxis[][SIZENUM], int CutPlace[][SIZENUM], double CutCost[][SIZENUM],
int leftSize[][SIZENUM], int rightSize[][SIZENUM],
int SetRankX[], int SetRankY[],
std::vector<std::vector<int>> &ans) {

	int SetId = SetHash[now_set];

	double size_x = size_list[size_id];
	double size_y = 1-size_x;

	int *order = new int[n];
	int now_n = 0;
	for(int i=0;i<n;i++)
	if(now_set&(1ll<<i)) {
		order[now_n] = i;
		now_n += 1;
	}

	double tot = 0;
	for(int i=0;i<now_n;i++)tot += weight[order[i]];

	if(now_n == 1) {
		delete[] order;
		return;
	}

	std::vector<int> cut;
	cut.push_back(CutAxis[SetId][size_id]);
	cut.push_back(CutPlace[SetId][size_id]);
	ans.push_back(cut);

	long long now_set1, now_set2;
	int axis = CutAxis[SetId][size_id];
	int place = CutPlace[SetId][size_id];
	if(axis==0)AxisRank(order, now_n, rank_x, SetRankX+(SetId*n));
	else AxisRank(order, now_n, rank_y, SetRankY+(SetId*n));
	now_set1 = now_set2 = 0;
	double tmp_count = 0;
	for(int i=0;i<=place;i++) {
		now_set1 += (1ll)<<order[i];
		tmp_count += weight[order[i]];
	}
	now_set2 = now_set-now_set1;
	double new_size_x1 = size_x;
	double new_size_y1 = size_y;
	double new_size_x2 = size_x;
	double new_size_y2 = size_y;
	if(axis==0){
		new_size_x1 = size_x*tmp_count/tot;
		new_size_x2 = size_x*(1-tmp_count/tot);
	}else{
		new_size_y1 = size_y*tmp_count/tot;
		new_size_y2 = size_y*(1-tmp_count/tot);
	}
//    int size_id1 = getSizeId(new_size_x1, new_size_y1, size_list);
//    int size_id2 = getSizeId(new_size_x2, new_size_y2, size_list);
    int size_id1 = leftSize[SetId][size_id];
    int size_id2 = rightSize[SetId][size_id];
	getAns(xy, sxy, rank_x, rank_y, weight, now_set1, n, size_id1, size_list, SetHash, SetCount, CutAxis, CutPlace, CutCost, leftSize, rightSize, SetRankX, SetRankY, ans);
	getAns(xy, sxy, rank_x, rank_y, weight, now_set2, n, size_id2, size_list, SetHash, SetCount, CutAxis, CutPlace, CutCost, leftSize, rightSize, SetRankX, SetRankY, ans);

	delete[] order;
	return;
}


std::vector<std::vector<int>> SearchForTree(std::vector<std::vector<double>> xy, std::vector<std::vector<double>> sxy,
std::vector<int> rank_x, std::vector<int> rank_y, std::vector<double> weight) {

//    double start = clock();

	int n = xy.size();

	std::vector<double> size_list;
//	size_list.push_back(0.9);
//	size_list.push_back(0.8);
//	size_list.push_back(0.667);
//	size_list.push_back(0.6);
//	size_list.push_back(0.5);
//	size_list.push_back(0.4);
//	size_list.push_back(0.333);
//	size_list.push_back(0.2);
//	size_list.push_back(0.1);
	for(int i=0;i<SIZENUM;i++) {
		size_list.push_back(1.0/(SIZENUM+1)*(i+1));
	}
	int mid = size_list.size()/2;
	int size_list_num = size_list.size();

	std::map<long long, int> SetHash;

	int m = 0;
	for(int i1=1;i1<=n;i1++)
	for(int i2=1;i2<=i1;i2++)
	for(int i3=1;i3<=i2;i3++)m += i3;

	int (*CutAxis)[SIZENUM] = new int[m][SIZENUM];
	int (*CutPlace)[SIZENUM] = new int[m][SIZENUM];
	double (*CutCost)[SIZENUM] = new double[m][SIZENUM];
	int (*leftSize)[SIZENUM] = new int[m][SIZENUM];
	int (*rightSize)[SIZENUM] = new int[m][SIZENUM];
	for(int i=0;i<m;i++)
	for(int j=0;j<SIZENUM;j++){
		CutAxis[i][j] = CutPlace[i][j] = -1;
		CutCost[i][j] = 0;
		leftSize[i][j] = rightSize[i][j] = -1;
	}

	int *SetRankX = new int[m*n];
	int *SetRankY = new int[m*n];
	for(int i=0;i<m;i++) {
	    SetRankX[i*n] = SetRankY[i*n] = -1;
	}

//    printf("%.4lf\n", (clock()-start)/CLOCKS_PER_SEC);

	int SetCount = 0;
	SearchForTreeDP(xy, sxy, rank_x, rank_y, weight, ((1ll)<<n)-1, n, mid, size_list, SetHash, SetCount, CutAxis, CutPlace, CutCost, leftSize, rightSize, SetRankX,SetRankY);

//    printf("%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf\n", time0, time1, time2, time3, time4, time5);

//    printf("%.4lf\n", (clock()-start)/CLOCKS_PER_SEC);

	std::vector<std::vector<int>> ans;
	getAns(xy, sxy, rank_x, rank_y, weight, ((1ll)<<n)-1, n, mid, size_list, SetHash, SetCount, CutAxis, CutPlace, CutCost, leftSize, rightSize, SetRankX, SetRankY, ans);

	delete[] CutAxis;
	delete[] CutPlace;
	delete[] CutCost;

	delete[] SetRankX;
	delete[] SetRankY;

//    printf("%.4lf\n", (clock()-start)/CLOCKS_PER_SEC);

	return ans;
}


#endif
