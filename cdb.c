#ifndef _GNU_SOURCE
#define _GNU_SOURCE         //asprintf
#endif

#include <stdio.h>          //print-
#include <stdlib.h>         //malloc etc
#include <string.h>         //strto- etc
#include <stdbool.h>        //bool
#include <fcntl.h>          //open
#include <errno.h>          //errno
#include <unistd.h>         //ftruncate
#include <sys/types.h>      //ftruncate
#include <sys/mman.h>       //mmap
#include <sys/time.h>       //gettimeofday
#include <stdint.h>     //uint8_t

#include "mpi.h"
#include "omp.h"

#define _FILE_OFFSET_BITS       64
#define die(...)                do { fprintf(stderr, ##__VA_ARGS__); exit(1); } while(0)
#define errstr()                strerror(errno)

typedef uint8_t                 byte;
typedef unsigned long           ulong;
typedef unsigned long long      ullong;

struct
{
    long                        size_in_mb;
    ullong                      total_bytes;
    char                        *prefix;
    int                         nitrs;
    bool                        new_file_per_write;
    bool                        first_file_setup;
    int                         nranks;
    int                         my_rank;
} params = {0};

void        init(int argc, char *argv[]);
void        parse_arguments(int argc, char *argv[]);
long        parse_long(const char* str);
void        print_help();
void        setup_file(int *fd_ptr, byte **file_ptr_ptr);
void        finalize_file(int *fd_ptr, byte **file_ptr_ptr);
double      do_write_benchmark(int *fd_ptr, byte **file_ptr_ptr);
double      get_unix_ms();
void        finalize();




int main(int argc, char *argv[])
{
    init(argc, argv);

    double sum_write_ms = 0;

    int i, nthreads;
    MPI_Comm_size(MPI_COMM_WORLD, &(params.nranks));

    #pragma omp parallel private(i)
    {
        nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int fd;
        byte *file_ptr;
        if (!params.new_file_per_write)
            setup_file(&fd, &file_ptr);

        double thread_sum_write_ms = 0.0f;
        for (i = 0; i < params.nitrs; i++)
        {
            if (!params.my_rank && !tid)
            {
                printf("Iteration #%d\n", i);
                fflush(stdout);
            }

            thread_sum_write_ms += do_write_benchmark(&fd, &file_ptr);
        }

        if (!params.my_rank && !tid)
            sum_write_ms += thread_sum_write_ms;

        if (!params.new_file_per_write)
            finalize_file(&fd, &file_ptr);
    }

    // double global_sum_write_ms = 0.0f;
    // MPI_Reduce(&sum_write_ms, &global_sum_write_ms, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (!params.my_rank)
    {
        printf("\nAverage disk write bandwidtch: %10.2f MB/s\n",
               (double) params.size_in_mb * nthreads * params.nranks * params.nitrs
               / (sum_write_ms / 1000));
        printf("Size in mb: %lu, NThreads: %d, NRanks: %d, Total write time in ms: %.2f, NItrs: %d\n",
               params.size_in_mb, nthreads, params.nranks, sum_write_ms, params.nitrs);
    }


    finalize();
    return 0;
}


double get_unix_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_usec + tv.tv_sec * 1e6) / 1000;
}

double do_write_benchmark(int *fd_ptr, byte **file_ptr_ptr)
{
    //get new file
    if (params.new_file_per_write)
        setup_file(fd_ptr, file_ptr_ptr);

    //create a memory from random data as source to write into file
    byte *mem_ptr = NULL;
    if ((mem_ptr = malloc(params.total_bytes)) == NULL)
        die("Could not allocate memory data source: %s\n", errstr());

    //write random stuff in the memory
    for(int i = 0; i < params.total_bytes-8; i+=8)
        mem_ptr[i] = 0xde;

    int tid = omp_get_thread_num();

    #pragma omp single
    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = 0.0f;
    if (!params.my_rank && !tid)
        t1 = get_unix_ms();

    //copy memory data source into file mapped in memory
    memcpy(*file_ptr_ptr, mem_ptr, params.total_bytes);

    //sync changes to file
    msync(*file_ptr_ptr, params.total_bytes, MS_SYNC);

    #pragma omp single
    MPI_Barrier(MPI_COMM_WORLD);

    double diff = 0.0f;
    if (!params.my_rank && !tid)
        diff = get_unix_ms() - t1;

    if (!params.my_rank && !tid)
    {
        printf("[R-%d-", params.my_rank);
        printf("T-%d] ", omp_get_thread_num());
        printf("Writing %llu bytes (%lu MB) took %.2f ms\n",
               params.total_bytes, params.size_in_mb, diff);
    }

    if (params.new_file_per_write)
        finalize_file(fd_ptr, file_ptr_ptr);

    free(mem_ptr);

    return diff;
}

void finalize_file(int *fd_ptr, byte **file_ptr_ptr)
{
    char *suffix = NULL;
    asprintf(&suffix, "%d.%d", params.my_rank, omp_get_thread_num());

    char *dest_file = NULL;
    asprintf(&dest_file, "%s.%s", params.prefix, suffix);

    if (*file_ptr_ptr)
        munmap(*file_ptr_ptr, params.total_bytes);

    if (*fd_ptr > 0)
        close(*fd_ptr);

    unlink(dest_file);
}

void setup_file(int *fd_ptr, byte **file_ptr_ptr)
{
    char *suffix = NULL;
    asprintf(&suffix, "%d.%d", params.my_rank, omp_get_thread_num());

    char *dest_file = NULL;
    asprintf(&dest_file, "%s.%s", params.prefix, suffix);

    //try to open the destination file
    if ((*fd_ptr =
                open(dest_file,
                     O_CREAT | O_EXCL | O_RDWR,
                     0600)
        ) <= 0)
    {
        die("Could not open file \"%s\": %s\n", dest_file, errstr());
    }

    //size the file
    ftruncate(*fd_ptr, params.total_bytes);

    //map file into memory
    if ((*file_ptr_ptr =
                mmap(0,
                     params.total_bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED,
                     *fd_ptr,
                     0)
        ) == NULL)
    {
        die("Could not map file into memory: %s\n", errstr());
    }
}

long parse_long(const char* str)
{
    char *endptr = NULL;
    long l = strtol(str, &endptr, 10);
    return strlen(endptr) == 0 ? l : -1;
}

void print_help()
{
    printf("Usage: cdb -s <size_in_mb> -d <dest_file>\n");
    printf("    Arguments:\n");
    printf("        -s, --size <mb>     : Specify the size in Mega Bytes to read/write\n");
    printf("        -d, --dest <file>   : Specify the destination file for read/write\n");
    printf("        -i, --itrs <n>      : Specify the number of iterations to avergae over (default: 10)\n");
    printf("        -n, --new           : Enable creating new file before every write (default: off)\n");
    printf("        -h, --help          : Show this help message\n");
    printf("\n");
}

void parse_arguments(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        const char *arg         = argv[i];
        const char *next_arg    = argv[i + 1];
        if (!strcmp(arg, "-s") || !strcmp(arg, "--size"))
        {
            if ((params.size_in_mb = parse_long(next_arg)) < 0)
            {
                fprintf(stderr, "Bad size: %s\n", argv[i]);
                print_help();
                exit(1);
            }
            else
            {
                i++;
                params.total_bytes = params.size_in_mb * 1024 * 1024;
            }
        }
        else if (!strcmp(arg, "-d") || !strcmp(arg, "--dest"))
        {
            asprintf(&params.prefix, "%s.cdb", next_arg);
            i++;
        }
        else if (!strcmp(arg, "-i") || !strcmp(arg, "--itr"))
        {
            if ((params.nitrs = parse_long(next_arg)) < 0)
            {
                fprintf(stderr, "Bad nitrs: %s\n", argv[i]);
                print_help();
                exit(1);
            }
            else
            {
                i++;
            }
        }
        else if (!strcmp(arg, "-n") || !strcmp(arg, "--new"))
        {
            printf("Warning: A new file will be created before each write.\n");
            printf("This might show lower performance even though the file creation is not timed.\n");
            params.new_file_per_write = true;
        }
        else if (!strcmp(arg, "-h") || !strcmp(arg, "--help"))
        {
            print_help();
            exit(1);
        }
        else
        {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_help();
            exit(1);
        }
    }

    bool pass = true;
    if (!(pass &= (params.size_in_mb > 0)))
        printf("Size not provided\n");

    if (!(pass &= (params.prefix != NULL)))
        printf("Destination file not provided\n");

    if (params.nitrs == 0)
        params.nitrs = 10;

    if (!pass)
    {
        print_help();
        exit(1);
    }
}

void init(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &(params.my_rank));

    parse_arguments(argc, argv);
}

void finalize()
{
    MPI_Finalize();
}
