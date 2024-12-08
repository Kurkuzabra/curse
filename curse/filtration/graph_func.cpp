#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <semaphore>
#include "combinatorics.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define EPSILON 0.000001
namespace py = pybind11;

std::binary_semaphore smphSignalThreadToMain{0};
std::counting_semaphore free_sem{0};
std::mutex inact_thr_mutex;

int ooo = 0, ooo_=0;

template<typename T>
inline bool vec_equal(std::vector<T>& vec1, std::vector<T>& vec2)
{
    bool flag = true;
    for (std::size_t i = 0; i < vec1.size(); i++)
    {
        if(vec1[i] != vec2[i])
        {
            flag = false;
            break;
        }
    }
    return flag;
}

double d(double* A_ptr, int A_shape, std::vector<int>& simplex_ptr, std::vector<int>& perm, double p)
{
    Subsequences subs = Subsequences(perm.size(), 2);
    double ds = std::numeric_limits<double>::min();

    if (std::fabs(p - (double)1) < EPSILON)
    {
        while(subs.next())
        {
            double norm_sqr = 0.0;
            const std::vector<int>& item = subs.get_subseq();
            for (std::size_t i = 0; i < item.size() - 1; i++)
            {
                norm_sqr += A_ptr[simplex_ptr[perm[item[i]]] * A_shape + simplex_ptr[perm[item[i + 1]]]];
            }
            ds = std::max(ds, norm_sqr);
        }
    }
    else if (p == std::numeric_limits<double>::infinity())
    {
        while(subs.next())
        {
            double norm_sqr = std::numeric_limits<float>::min();
            const std::vector<int>& item = subs.get_subseq();
            for (std::size_t i = 0; i < item.size() - 1; i++)
            {
                norm_sqr = std::max(norm_sqr, A_ptr[simplex_ptr[perm[item[i]]] * A_shape + simplex_ptr[perm[item[i + 1]]]]);
            }
            ds = std::max(ds, norm_sqr);
        }
    }
    else
    {
        while(subs.next())
        {
            double norm_sqr = 0.0;
            const std::vector<int>& item = subs.get_subseq();
            for (std::size_t i = 0; i < item.size() - 1; i++)
            {
                norm_sqr += std::pow(A_ptr[simplex_ptr[perm[item[i]]] * A_shape + simplex_ptr[perm[item[i + 1]]]], p);
            }
            ds = std::max(ds, std::pow(norm_sqr, p));
        }
    }
    return ds;
}

double d(double* A_ptr, int A_shape, int* simplex_ptr, std::vector<int>& perm, double p)
{
    Subsequences subs = Subsequences(perm.size(), 2);
    double ds = std::numeric_limits<double>::min();

    if (std::fabs(p - (double)1) < EPSILON)
    {
        while(subs.next())
        {
            double norm_sqr = 0.0;
            const std::vector<int>& item = subs.get_subseq();
            for (std::size_t i = 0; i < item.size() - 1; i++)
            {
                norm_sqr += A_ptr[simplex_ptr[perm[item[i]]] * A_shape + simplex_ptr[perm[item[i + 1]]]];
            }
            ds = std::max(ds, norm_sqr);
        }
    }
    else if (p == std::numeric_limits<double>::infinity())
    {
        while(subs.next())
        {
            double norm_sqr = std::numeric_limits<float>::min();
            const std::vector<int>& item = subs.get_subseq();
            for (std::size_t i = 0; i < item.size() - 1; i++)
            {
                norm_sqr = std::max(norm_sqr, A_ptr[simplex_ptr[perm[item[i]]] * A_shape + simplex_ptr[perm[item[i + 1]]]]);
            }
            ds = std::max(ds, norm_sqr);
        }
    }
    else
    {
        while(subs.next())
        {
            double norm_sqr = 0.0;
            const std::vector<int>& item = subs.get_subseq();
            for (std::size_t i = 0; i < item.size() - 1; i++)
            {
                norm_sqr += std::pow(A_ptr[simplex_ptr[perm[item[i]]] * A_shape + simplex_ptr[perm[item[i + 1]]]], p);
            }
            ds = std::max(ds, std::pow(norm_sqr, 1.0 / p));
        }
    }
    return ds;
}

double f_part(double* A_ptr, int A_sz, int* simplex_ptr, std::vector<int>& perm_, std::vector<int>& last_perm_, double p,
              std::vector<int>& inactive_threads, int num_thread)
{
    std::vector<int> perm(perm_.size());
    std::vector<int> last_perm(last_perm_.size());
    for (std::size_t i = 0; i < perm.size(); i++)
    {
        perm[i] = perm_[i];
        last_perm[i] = last_perm_[i];
    }
    smphSignalThreadToMain.release();

    double fs = std::numeric_limits<double>::max();

    for (std::size_t i = 0; i < perm.size(); i++) perm[i] = i;

    do
    {
        for (std::size_t i = 0; i < perm.size() / 2; i++)
        {
            if (perm[i] > perm[perm.size() - i - 1])
            {
                continue;
            }
        }

        fs = std::min(fs, d(A_ptr, A_sz, simplex_ptr, std::ref(perm), p));
    }
    while (std::next_permutation(perm.begin(), perm.end()) && !vec_equal(perm, last_perm));

    std::unique_lock<std::mutex> thr_lock{inact_thr_mutex, std::defer_lock};
    thr_lock.lock();

    inactive_threads.push_back(num_thread);

    thr_lock.unlock();

    return fs;

}

void perm_start_with(std::vector<int>& perm, const std::vector<int>& start)
{
    int ins_pos = start.size();
    for (std::size_t i = 0; i < start.size(); i++) perm[i] = start[i];
    for(std::size_t i = 0; i < perm.size(); i++)
    {
        for (std::size_t j = 0; j < start.size(); j++)
        {
            if (start[j] == (int)i) goto perm_start_with_val_exists; 
        }
        perm[ins_pos] = i;
        ins_pos++;
        perm_start_with_val_exists:;
    }
}

double f_single_thread_(double* A_ptr, int A_sz, std::vector<int> simplex, double p)
{
    std::vector<int> perm(simplex.size());
    double fs = std::numeric_limits<double>::max();

    for (std::size_t i = 0; i < perm.size(); i++) perm[i] = i;
    do
    {
        fs = std::min(fs, d(A_ptr, A_sz, simplex, perm, p));
    } while(std::next_permutation(perm.begin(), perm.end()));

    return fs;   
}

void f_single_thread_part_(std::array<double, 3>& res_ptr, double* A_ptr, int A_sz, std::vector<int> simplex)
{
    smphSignalThreadToMain.release();
    res_ptr[0] = f_single_thread_(A_ptr, A_sz, simplex, 1);
    res_ptr[1] = f_single_thread_(A_ptr, A_sz, simplex, 2);
    res_ptr[2] = f_single_thread_(A_ptr, A_sz, simplex, std::numeric_limits<double>::infinity());
    free_sem.release();
}


double f_single_thread(const py::array_t<double>& A, const py::array_t<int>& simplex, double p)
{
    py::buffer_info A_arr = A.request();
    py::buffer_info simplex_arr = simplex.request();
    int simplex_sz = simplex_arr.shape[0];
    int A_sz = A_arr.shape[0];

    double* A_ptr = static_cast<double *>(A_arr.ptr);
    int* simplex_ptr = static_cast<int *>(simplex_arr.ptr);

    std::vector<int> perm(simplex_sz);
    double fs = std::numeric_limits<double>::max();

    for (std::size_t i = 0; i < perm.size(); i++) perm[i] = i;

    do
    {
        fs = std::min(fs, d(A_ptr, A_sz, simplex_ptr, perm, p));
        // ooo_++;
    } while(std::next_permutation(perm.begin(), perm.end()));

    // std::cout << ooo_ << std::endl;

    return fs;   
}

template <class It>
double f_mult_part
                (
                    double* A_ptr, int A_sz, int* simplex_ptr, int simplex_sz, int perm_beg_part,
                    It& begin, It& end, double p, std::vector<int>& inactive_threads, int num_thread
                )
{
    std::vector<int> perm(simplex_sz);

    for (int i = 0; i < simplex_sz; i++)
    {
        // std::cout << *(begin + i) << std::endl;
        perm[i] = *(begin + i);
    }
    smphSignalThreadToMain.release();

    double fs = std::numeric_limits<double>::max();

    do
    {
        for (std::size_t i = 0; i < perm.size() / 2; i++)
        {
            if (perm[i] > perm[perm.size() - i - 1])
            {
                continue;
            }
        }

        fs = std::min(fs, d(A_ptr, A_sz, simplex_ptr, std::ref(perm), p));
    }
    while (std::next_permutation(perm.begin() + perm_beg_part, perm.end()));

    std::unique_lock<std::mutex> thr_lock{inact_thr_mutex, std::defer_lock};
    thr_lock.lock();
    inactive_threads.push_back(num_thread);
    free_sem.release();
    thr_lock.unlock();

    return fs;

}

class f_class
{
    double* A_ptr;
    int A_sz;
    int* simplex_ptr;
    int simplex_sz;
    int perm_beg_part;
    double p;
    double fs;
    std::vector<int>& thr_free_queue;
    std::vector<int>& func_initialised;
    std::vector<std::future<double>>& func_res;
    std::unique_lock<std::mutex> thr_lock;

public:
    f_class
    (
        double* _A_ptr, int _A_sz, int* _simplex_ptr, int _simplex_sz, int _perm_beg_part, double _p,
        std::vector<int>& _thr_free_queue, std::vector<int>& _func_initialised, std::vector<std::future<double>>& _func_res
    ) : A_ptr(_A_ptr), A_sz(_A_sz), simplex_ptr(_simplex_ptr), simplex_sz(_simplex_sz), p(_p), perm_beg_part(_perm_beg_part),
        thr_free_queue(_thr_free_queue), func_initialised(_func_initialised), func_res(_func_res),
        fs(std::numeric_limits<double>::max())
    {
        thr_lock = std::unique_lock<std::mutex>(inact_thr_mutex, std::defer_lock);
    }
    
    template <class It>
    bool operator()(It first, It last)
    {

        for (std::size_t i = 0; i < thr_free_queue.size(); i++)
        {
            free_sem.release(); 
        }

        free_sem.acquire();
        thr_lock.lock();
        int last_thr = thr_free_queue[thr_free_queue.size() - 1];
        thr_free_queue.pop_back();
        thr_lock.unlock();        

        if (func_initialised[last_thr])
        {
            fs = std::min(fs, func_res[last_thr].get());
        }
        else 
        {
            func_initialised[last_thr] = 1;
        }
        func_res[last_thr] = std::async
        (
            f_mult_part<It>, A_ptr, A_sz, simplex_ptr, simplex_sz, perm_beg_part,                            
            std::ref(first), std::ref(last), p, std::ref(thr_free_queue), last_thr
        );
        smphSignalThreadToMain.acquire();

        return false;
    }

    operator double() 
    {
        return fs;
    }

};

double f_multithread(const py::array_t<double>& A, const py::array_t<int>& simplex, double p, int num_threads = 1)
{
    py::buffer_info A_arr = A.request();
    py::buffer_info simplex_arr = simplex.request();
    int simplex_sz = simplex_arr.shape[0];
    int A_sz = A_arr.shape[0];

    double* A_ptr = static_cast<double *>(A_arr.ptr);
    int* simplex_ptr = static_cast<int *>(simplex_arr.ptr);

    int perm_sz = A_sz;
    long long task_cnt = perm_sz;
    int perm_beg_sz = 0;
    while (num_threads > task_cnt && perm_sz > 1 && perm_beg_sz < simplex_sz - 1)
    {
        perm_sz--;
        task_cnt *= perm_sz;
        perm_beg_sz++;
    }
    std::vector<int> perm(simplex_sz);
    std::vector<std::future<double>> func_res(num_threads);
    std::vector<int> thr_free_queue(num_threads);
    std::vector<int> func_initialised(num_threads);



    for (int i = 0; i < simplex_sz; i++) perm[i] = i;
    for (std::size_t i = 0; i < thr_free_queue.size(); i++)
    {
        thr_free_queue[i] = i;
        func_initialised[i] = 0;
    }

    double fs = for_each_permutation(
                                        perm.begin(),
                                        perm.begin() + perm_beg_sz,
                                        perm.end(),
                                        f_class
                                        (
                                            A_ptr, A_sz, simplex_ptr, simplex_sz, perm_beg_sz, p, std::ref(thr_free_queue),
                                            std::ref(func_initialised), std::ref(func_res)
                                        )
                                    );

    for (std::size_t i = 0; i < func_res.size(); i++)
    {
        if (func_initialised[i])
        {
            fs = std::min(fs, func_res[i].get());
        }
    }

    return fs;   
}

py::array_t<double> filtrate(const py::array_t<double>& A, int simplex_sz, int num_threads = 1)
{
    py::buffer_info A_arr = A.request();
    int A_sz = A_arr.shape[0];
    double* A_ptr = static_cast<double *>(A_arr.ptr);
    std::vector<std::array<double, 3>> result(nChoosek(A_sz, simplex_sz));
    Combinations comb(A_sz, simplex_sz);
    long long i = 0;
    if (num_threads == 1)
    {
        do
        {
            const std::vector<int>& simplex = comb.get_comb();
            // result.push_back(std::array<double, 3>());
            result[i][0] = f_single_thread_(A_ptr, A_sz, simplex, 1);
            result[i][1] = f_single_thread_(A_ptr, A_sz, simplex, 2.0);
            result[i][2] = f_single_thread_(A_ptr, A_sz, simplex, std::numeric_limits<double>::infinity());
            i++;
        } while(comb.next());
    }
    else
    {
        std::vector<std::future<double>> func_res(num_threads);
        std::vector<int> thr_free_queue(num_threads);
        std::vector<int> func_initialised(num_threads);

        for (int i = 0; i < num_threads; i++)
        {
            free_sem.release();
            thr_free_queue[i] = i;
            func_initialised[i] = 0;
        }
        do
        {
            const std::vector<int>& simplex = comb.get_comb();

            free_sem.acquire();

            std::thread thr(f_single_thread_part_, std::ref(result[i]), A_ptr, A_sz, simplex);
            thr.detach();

            smphSignalThreadToMain.acquire();
            i++;
        } while(comb.next());

        for (int i = 0; i < num_threads; i++)
        {
            free_sem.acquire();
        }
    }
    return py::array_t<double>(py::cast(result));
}

PYBIND11_MODULE(graph_func, m) {
    m.doc() = "pybind11 graph filtration";
    m.def("f_multithread", &f_multithread, "A filtration over graph");
    m.def("f_single_thread", &f_single_thread, "A filtration over graph");
    m.def("filtrate", &filtrate, "filter complex", py::arg("A"), py::arg("n"), py::arg("threads"));
}
