#ifndef _OPTIMIZE_H
#define _OPTIMIZE_H

#include "base.h"

std::vector<double> check_cost_type(
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const std::string &type,
const std::vector<bool> &_consider) {

    int N = _grid_asses.size();
    Optimizer op(0);
    std::vector<double> _tmp_asses = op.checkCostForAll(_ori_embedded, _grid_asses, _cluster_labels, type, 1, 0, _consider);
    std::vector<double> new_cost(3, 0);
    for(int i=0;i<3;i++)new_cost[i] = _tmp_asses[i+N];
    return new_cost;
}

std::vector<double> solve_op(
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const std::string &type,
double alpha, double beta,
bool alter, std::vector<double> alter_best,
int maxit, int maxit2,
int swap_cnt, int choose_k,
const std::vector<bool> &_consider,
const std::vector<bool> &_change,
const std::vector<int> &_other_label) {

    int N = _grid_asses.size();

    std::vector<int> _new_grid_asses(_grid_asses);
    std::vector<int> _new_grid_asses2(_grid_asses);

    std::vector<int> ans(_grid_asses);
    std::vector<double> new_cost(3, 2147483647);
    std::vector<double> best_cost(3, 2147483647);

    std::vector<double> _tmp_asses;
    std::vector<double> ret(N+3, 0);

    Optimizer op(0);
    if(maxit > 0){
        _tmp_asses = op.optimizeBA(_ori_embedded, _grid_asses, _cluster_labels, _change, type, alpha, beta, alter,
                                     alter_best, maxit, _consider, _other_label);
        for(int i=0;i<N;i++)_new_grid_asses[i] = int(_tmp_asses[i]+0.5);
        for(int i=0;i<3;i++)new_cost[i] = _tmp_asses[i+N];

        ans = _new_grid_asses;
        best_cost = new_cost;
    }

    if(maxit2 > 0){
        int seed = 10;

        bool innerBiMatch = true;
        if(type=="Edges")innerBiMatch = false;

        _tmp_asses = op.optimizeSwap(_ori_embedded, _new_grid_asses, _cluster_labels, _change, type, alpha, beta,
                                       maxit2, choose_k, seed, innerBiMatch, swap_cnt, _consider, _other_label);
        for(int i=0;i<N;i++)_new_grid_asses2[i] = int(_tmp_asses[i]+0.5);
        for(int i=0;i<3;i++)new_cost[i] = _tmp_asses[i+N];

        ans = _new_grid_asses2;
        best_cost = new_cost;
    }

    if((type != "Global") && (maxit2 == 0)){
        _new_grid_asses2 = op.optimizeInnerCluster(_ori_embedded, _new_grid_asses, _cluster_labels, _change, _consider);
        new_cost = std::vector<double>(3, -1);

        ans = _new_grid_asses2;
        best_cost = new_cost;
    }

    for(int i=0;i<N;i++)ret[i] = ans[i];
    for(int i=0;i<3;i++)ret[i+N] = new_cost[i];
    return ret;
}

std::vector<double> grid_op(
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<std::vector<double>> &_grid_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const bool useGlobal,
const bool useLocal,
const std::string &type,
int maxit, int maxit2,
bool only_compact,
int swap_cnt, int choose_k,
const std::vector<bool> &_consider,
const std::vector<bool> &_change,
const std::vector<int> &_other_label) {

    int N = _grid_asses.size();
    int compact_it = 3;
    int global_it = 3;
    bool alter = true;

    std::vector<int> ori_grid_asses(_grid_asses);

    std::vector<int> ans(_grid_asses);
    std::vector<double> new_cost(3, 2147483647);

    std::vector<double> alter_best(2, 0);
    alter_best.push_back(1); alter_best.push_back(1);

    std::vector<double> _tmp_asses;
    std::vector<double> ret(N+3, 0);

    if(useGlobal){
        if(alter){
            std::vector<int> _grid_asses1(ori_grid_asses);
            std::vector<double> new_cost1 = check_cost_type(_grid_embedded, _grid_asses1, _cluster_labels, "Global", _consider);
            _tmp_asses = solve_op(_grid_embedded, ans, _cluster_labels, "Global", 0, 1, false, alter_best, compact_it, 0,
                            swap_cnt, choose_k, _consider, _change, _other_label);
            std::vector<int> _grid_asses2(N);
            std::vector<double> new_cost2(3);
            for(int i=0;i<N;i++)_grid_asses2[i] = int(_tmp_asses[i]+0.5);
            for(int i=0;i<3;i++)new_cost2[i] = _tmp_asses[i+N];

            alter_best[0] = std::min(new_cost1[0], new_cost2[0]);
            alter_best[2] = std::max(new_cost1[0], new_cost2[0]) - std::min(new_cost1[0], new_cost2[0]);
            alter_best[1] = std::min(new_cost1[1], new_cost2[1]);
            alter_best[3] = std::max(new_cost1[1], new_cost2[1]) - std::min(new_cost1[1], new_cost2[1]);

            if(only_compact){
                ans = _grid_asses2;
                new_cost = new_cost2;
            }else if((std::abs(alter_best[3]) < 0.0000001) || (std::abs(alter_best[2]) < 0.000001)) {
                ans = _grid_asses1;
                new_cost = new_cost1;
            }else {
                _tmp_asses = solve_op(_grid_embedded, ans, _cluster_labels, "Global", 0, 0.5, true, alter_best, global_it, 0,
                            swap_cnt, choose_k, _consider, _change, _other_label);
                for(int i=0;i<N;i++)ans[i] = int(_tmp_asses[i]+0.5);
                for(int i=0;i<3;i++)new_cost[i] = _tmp_asses[i+N];
            }
        }else {
            _tmp_asses = solve_op(_grid_embedded, ans, _cluster_labels, "Global", 0, 0.5, true, alter_best, 1, 0,
                            swap_cnt, choose_k, _consider, _change, _other_label);
            for(int i=0;i<N;i++)ans[i] = int(_tmp_asses[i]+0.5);
            for(int i=0;i<3;i++)new_cost[i] = _tmp_asses[i+N];
        }
    }

    if(useLocal){
        std::vector<double> _tmp_asses = solve_op(_ori_embedded, ans, _cluster_labels, type, 1, 0, false, alter_best, maxit, maxit2,
                            swap_cnt, choose_k, _consider, _change, _other_label);
        for(int i=0;i<N;i++)ans[i] = int(_tmp_asses[i]+0.5);
        for(int i=0;i<3;i++)new_cost[i] = _tmp_asses[i+N];
    }

    for(int i=0;i<N;i++)ret[i] = ans[i];
    for(int i=0;i<3;i++)ret[i+N] = new_cost[i];
    return ret;
}

std::vector<int> grid_assign_partition(
const int partitions,
const std::vector<int> &_partition,
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels) {

    int N = _partition.size();
    int num = _cluster_labels.size();
    int square_len = ceil(sqrt(N));
    std::vector<int> ret(_grid_asses);

    // omp_set_nested(true);
    #pragma omp parallel for num_threads(partitions)
    for(int pi=0;pi<partitions;pi++) {
        std::vector<int> _other_label;
        std::vector<bool> _change(N, false);

        int min_x = square_len-1, max_x = 0, min_y = square_len-1, max_y = 0;
        int cnt = 0;
        for(int i=0;i<N;i++) {
            if(_partition[i] == pi) {
                _change[i] = true;
                cnt += 1;
                int x = i/square_len;
                int y = i%square_len;
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }

        printf("start assign %d %d\n", pi, cnt);

        Optimizer op(0);
        std::vector<double> alter_best(4, 0);

        // std::vector<double> tmp_asses = op.optimizeBA(_ori_embedded, _grid_asses, _cluster_labels, _change, "Global",
        //                     0, 0, false, alter_best, 1, _change, _other_label);
        // for(int i=0;i<N;i++)
        //     if(_partition[i] == pi)ret[i] = int(tmp_asses[i]+0.5);

        int small_len = std::max(max_x-min_x+1, max_y-min_y+1);
        if(small_len<=0)continue;
        min_x = std::min(square_len-small_len, min_x);
        min_y = std::min(square_len-small_len, min_y);

        int small_N = small_len*small_len;
        std::vector<bool> small_change(small_N, false);
        std::vector<std::vector<double>> small_ori_embedded(small_N, std::vector<double>(2, 0));
        std::vector<int> small_grid_asses(small_N, 0);
        std::vector<int> small_cluster_labels;
        std::vector<int> small_true_id;
        std::vector<int> small_other_label;

        int small_num = 0;
        for(int x=min_x; x<min_x+small_len; x++) {
            int x0 = x - min_x;
            for(int y=min_y; y<min_y+small_len; y++) {
                int y0 = y - min_y;
                int gid = x*square_len+y;
                int gid0 = x0*small_len+y0;
                if(_partition[gid] == pi) {
                    small_change[gid0] = true;
                }
                if(_grid_asses[gid]<num) {
                    small_cluster_labels.push_back(_cluster_labels[_grid_asses[gid]]);
                    small_true_id.push_back(_grid_asses[gid]);
                    small_grid_asses[gid0] = small_num;
                    small_num += 1;
                }
            }
        }

        int small_num2 = small_num;
        for(int x=min_x; x<min_x+small_len; x++) {
            int x0 = x - min_x;
            for(int y=min_y; y<min_y+small_len; y++) {
                int y0 = y - min_y;
                int gid = x*square_len+y;
                int gid0 = x0*small_len+y0;
                if(_grid_asses[gid]>=num) {
                    small_true_id.push_back(_grid_asses[gid]);
                    small_grid_asses[gid0] = small_num2;
                    small_num2 += 1;
                }
                small_ori_embedded[small_grid_asses[gid0]] = _ori_embedded[_grid_asses[gid]];
                small_ori_embedded[small_grid_asses[gid0]][0] -= 1.0*min_x/square_len;
                small_ori_embedded[small_grid_asses[gid0]][1] -= 1.0*min_y/square_len;
                small_ori_embedded[small_grid_asses[gid0]][0] /= 1.0*small_len/square_len;
                small_ori_embedded[small_grid_asses[gid0]][1] /= 1.0*small_len/square_len;
            }
        }

        std::vector<double> tmp_small_asses = op.optimizeBA(small_ori_embedded, small_grid_asses, small_cluster_labels, small_change, "Global",
                            0, 0, false, alter_best, 1, small_change, small_other_label);
        for(int x=min_x; x<min_x+small_len; x++) {
            int x0 = x - min_x;
            for(int y=min_y; y<min_y+small_len; y++) {
                int y0 = y - min_y;
                int gid = x*square_len+y;
                int gid0 = x0*small_len+y0;
                if(_partition[gid] == pi)ret[gid] = small_true_id[int(tmp_small_asses[gid0]+0.5)];
            }
        }
    }

    return ret;
}

std::vector<int> grid_op_partition(
const int partitions,
const std::vector<int> &_partition,
const std::vector<std::vector<double>> &_ori_embedded,
const std::vector<std::vector<double>> &_grid_embedded,
const std::vector<int> &_grid_asses,
const std::vector<int> &_cluster_labels,
const bool useGlobal,
const bool useLocal,
const std::string &type,
int maxit, int maxit2,
bool only_compact,
int swap_cnt, int choose_k,
std::vector<int> _other_label) {

    int N = _partition.size();
    int num = _cluster_labels.size();
    int square_len = ceil(sqrt(N));
    std::vector<int> ret(_grid_asses);

    int maxLabel = 0;
    for(int i=0;i<num;i++)maxLabel = std::max(maxLabel, _cluster_labels[i]+1);
    std::vector<int> _new_other_label(0);
    for(int i=0;i<_other_label.size();i++)
    if(_other_label[i]<maxLabel)
        _new_other_label.push_back(_other_label[i]);
    _other_label = _new_other_label;

    // omp_set_nested(true);
    #pragma omp parallel for num_threads(partitions)
    for(int pi=0;pi<partitions;pi++) {
        bool flag = false;
        int* label_cnt = new int[maxLabel];
        for(int i=0;i<maxLabel;i++)label_cnt[i] = 0;

        int p_cnt = 0;
        double start = clock();
        std::vector<bool> _change(N, false);

        int min_x = square_len-1, max_x = 0, min_y = square_len-1, max_y = 0;

        for(int i=0;i<N;i++)
        if(_partition[i] == pi){
            _change[i] = true;
            if(_grid_asses[i]<num){
                p_cnt += 1;
                label_cnt[_cluster_labels[_grid_asses[i]]] += 1;
                int x = i/square_len;
                int y = i%square_len;
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }
        for(int i=0;i<maxLabel;i++)
        if((label_cnt[i]>0)&&(label_cnt[i]<p_cnt))flag = true;
        
        if(flag) {
            // std::vector<double> tmp_asses = grid_op(_ori_embedded, _grid_embedded, _grid_asses, _cluster_labels, useGlobal, useLocal, type, maxit, maxit2,
            //             only_compact, swap_cnt, choose_k, _change, _change, _other_label);

            // for(int i=0;i<N;i++)
            //     if(_partition[i] == pi)ret[i] = int(tmp_asses[i]+0.5);

            int small_len = std::max(max_x-min_x+1, max_y-min_y+1);
            if(small_len<=0)continue;
            min_x = std::min(square_len-small_len, min_x);
            min_y = std::min(square_len-small_len, min_y);

            int small_N = small_len*small_len;
            std::vector<bool> small_change(small_N, false);
            std::vector<std::vector<double>> small_ori_embedded(small_N, std::vector<double>(2, 0));
            std::vector<std::vector<double>> small_grid_embedded(small_N, std::vector<double>(2, 0));
            std::vector<int> small_grid_asses(small_N, 0);
            std::vector<int> small_cluster_labels;
            std::vector<int> small_true_id;
            std::vector<int> small_other_label;

            int small_num = 0;
            for(int x=min_x; x<min_x+small_len; x++) {
                int x0 = x - min_x;
                for(int y=min_y; y<min_y+small_len; y++) {
                    int y0 = y - min_y;
                    int gid = x*square_len+y;
                    int gid0 = x0*small_len+y0;
                    if(_partition[gid] == pi) {
                        small_change[gid0] = true;
                    }
                    if(_grid_asses[gid]<num) {
                        small_cluster_labels.push_back(_cluster_labels[_grid_asses[gid]]);
                        small_true_id.push_back(_grid_asses[gid]);
                        small_grid_asses[gid0] = small_num;
                        small_num += 1;
                    }
                }
            }
            int small_num2 = small_num;
            for(int x=min_x; x<min_x+small_len; x++) {
                int x0 = x - min_x;
                for(int y=min_y; y<min_y+small_len; y++) {
                    int y0 = y - min_y;
                    int gid = x*square_len+y;
                    int gid0 = x0*small_len+y0;
                    if(_grid_asses[gid]>=num) {
                        small_true_id.push_back(_grid_asses[gid]);
                        small_grid_asses[gid0] = small_num2;
                        small_num2 += 1;
                    }
                    small_ori_embedded[small_grid_asses[gid0]] = _ori_embedded[_grid_asses[gid]];
                    small_ori_embedded[small_grid_asses[gid0]][0] -= 1.0*min_x/square_len;
                    small_ori_embedded[small_grid_asses[gid0]][1] -= 1.0*min_y/square_len;
                    small_ori_embedded[small_grid_asses[gid0]][0] /= 1.0*small_len/square_len;
                    small_ori_embedded[small_grid_asses[gid0]][1] /= 1.0*small_len/square_len;
                    small_grid_embedded[small_grid_asses[gid0]] = _grid_embedded[_grid_asses[gid]];
                    small_grid_embedded[small_grid_asses[gid0]][0] -= 1.0*min_x/square_len;
                    small_grid_embedded[small_grid_asses[gid0]][1] -= 1.0*min_y/square_len;
                    small_grid_embedded[small_grid_asses[gid0]][0] /= 1.0*small_len/square_len;
                    small_grid_embedded[small_grid_asses[gid0]][1] /= 1.0*small_len/square_len;
                }
            }
            for(int i=0;i<_other_label.size();i++)
            if(label_cnt[_other_label[i]]>0){
                small_other_label.push_back(_other_label[i]);
            }

            std::vector<double> tmp_small_asses = grid_op(small_ori_embedded, small_grid_embedded, small_grid_asses, small_cluster_labels, useGlobal, useLocal, type, maxit, maxit2,
                        only_compact, swap_cnt, choose_k, small_change, small_change, small_other_label);
            for(int x=min_x; x<min_x+small_len; x++) {
                int x0 = x - min_x;
                for(int y=min_y; y<min_y+small_len; y++) {
                    int y0 = y - min_y;
                    int gid = x*square_len+y;
                    int gid0 = x0*small_len+y0;
                    if(_partition[gid] == pi)ret[gid] = small_true_id[int(tmp_small_asses[gid0]+0.5)];
                }
            }
        }

        // printf("optimize %d time: %.2lf\n", pi, (clock()-start)/CLOCKS_PER_SEC);
        
        delete[] label_cnt;
    }

    return ret;
}

#endif