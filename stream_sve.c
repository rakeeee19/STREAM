/*
 * STREAM ARM SVE variant.
 *
 * Kernels:
 *   Write, Read, Copy, Scale, Add, Triad
 *
 * Build on SVE-capable ARM Linux:
 *   gcc -O3 -fopenmp -march=armv8-a+sve stream_sve.c -o stream_sve.exe
 *
 * Select kernels at runtime:
 *   ./stream_sve.exe all
 *   ./stream_sve.exe wr_rd
 *   ./stream_sve.exe write
 *   ./stream_sve.exe read
 *   ./stream_sve.exe copy
 *   ./stream_sve.exe scale
 *   ./stream_sve.exe add
 *   ./stream_sve.exe triad
 */

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#define HAVE_SVE 1
#else
#define HAVE_SVE 0
#endif

#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 10000000
#endif

#ifdef NTIMES
#if NTIMES <= 1
#define NTIMES 10
#endif
#endif
#ifndef NTIMES
#define NTIMES 10
#endif

#ifndef OFFSET
#define OFFSET 0
#endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

#define HLINE "-------------------------------------------------------------\n"
#define NUM_KERNELS 6
#define K_WRITE 0
#define K_READ 1
#define K_COPY 2
#define K_SCALE 3
#define K_ADD 4
#define K_TRIAD 5

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif
#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

static STREAM_TYPE a[STREAM_ARRAY_SIZE + OFFSET];
static STREAM_TYPE b[STREAM_ARRAY_SIZE + OFFSET];
static STREAM_TYPE c[STREAM_ARRAY_SIZE + OFFSET];
static volatile STREAM_TYPE read_sink;

static const char *labels[NUM_KERNELS] = {
    "Write:     ", "Read:      ", "Copy:      ",
    "Scale:     ", "Add:       ", "Triad:     ",
};

static const double bytes[NUM_KERNELS] = {
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
};

static double mysecond(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

static int parse_mode(int argc, char **argv, int enabled[NUM_KERNELS]) {
  int i;

  if (argc <= 1 || strcmp(argv[1], "all") == 0) {
    for (i = 0; i < NUM_KERNELS; i++) {
      enabled[i] = 1;
    }
    return 0;
  }

  if (strcmp(argv[1], "wr_rd") == 0 || strcmp(argv[1], "read_write") == 0) {
    enabled[K_WRITE] = 1;
    enabled[K_READ] = 1;
    return 0;
  }

  if (strcmp(argv[1], "write") == 0) {
    enabled[K_WRITE] = 1;
  } else if (strcmp(argv[1], "read") == 0) {
    enabled[K_READ] = 1;
  } else if (strcmp(argv[1], "copy") == 0) {
    enabled[K_COPY] = 1;
  } else if (strcmp(argv[1], "scale") == 0) {
    enabled[K_SCALE] = 1;
  } else if (strcmp(argv[1], "add") == 0) {
    enabled[K_ADD] = 1;
  } else if (strcmp(argv[1], "triad") == 0) {
    enabled[K_TRIAD] = 1;
  } else {
    return -1;
  }

  return 0;
}

static int checktick(void) {
  enum { M = 20 };
  int i, min_delta, delta;
  double t1, t2, timesfound[M];

  for (i = 0; i < M; i++) {
    t1 = mysecond();
    while (((t2 = mysecond()) - t1) < 1.0E-6) {
    }
    timesfound[i] = t1 = t2;
  }

  min_delta = 1000000;
  for (i = 1; i < M; i++) {
    delta = (int)(1.0E6 * (timesfound[i] - timesfound[i - 1]));
    min_delta = MIN(min_delta, MAX(delta, 0));
  }

  return min_delta;
}

#if HAVE_SVE
static ssize_t sve_block_elems(void) { return (ssize_t)svcntd() * 16; }

static void kernel_write(STREAM_TYPE value) {
  ssize_t base, j, end;
  const ssize_t block = sve_block_elems();
  const svfloat64_t v = svdup_f64(value);

#pragma omp parallel for private(j, end) schedule(static)
  for (base = 0; base < STREAM_ARRAY_SIZE; base += block) {
    end = MIN(base + block, STREAM_ARRAY_SIZE);
    for (j = base; j < end;) {
      svbool_t pg = svwhilelt_b64(j, end);
      svst1_f64(pg, &a[j], v);
      j += svcntd();
    }
  }
}

static STREAM_TYPE kernel_read(void) {
  STREAM_TYPE total = 0.0;

#pragma omp parallel
  {
    ssize_t base, j, end;
    const ssize_t block = sve_block_elems();
    svfloat64_t acc = svdup_f64(0.0);

#pragma omp for schedule(static)
    for (base = 0; base < STREAM_ARRAY_SIZE; base += block) {
      end = MIN(base + block, STREAM_ARRAY_SIZE);
      for (j = base; j < end;) {
        svbool_t pg = svwhilelt_b64(j, end);
        acc = svadd_f64_m(pg, acc, svld1_f64(pg, &a[j]));
        j += svcntd();
      }
    }

#pragma omp atomic
    total += svaddv_f64(svptrue_b64(), acc);
  }

  return total;
}

static void kernel_copy(void) {
  ssize_t base, j, end;
  const ssize_t block = sve_block_elems();

#pragma omp parallel for private(j, end) schedule(static)
  for (base = 0; base < STREAM_ARRAY_SIZE; base += block) {
    end = MIN(base + block, STREAM_ARRAY_SIZE);
    for (j = base; j < end;) {
      svbool_t pg = svwhilelt_b64(j, end);
      svst1_f64(pg, &c[j], svld1_f64(pg, &a[j]));
      j += svcntd();
    }
  }
}

static void kernel_scale(STREAM_TYPE scalar) {
  ssize_t base, j, end;
  const ssize_t block = sve_block_elems();
  const svfloat64_t s = svdup_f64(scalar);

#pragma omp parallel for private(j, end) schedule(static)
  for (base = 0; base < STREAM_ARRAY_SIZE; base += block) {
    end = MIN(base + block, STREAM_ARRAY_SIZE);
    for (j = base; j < end;) {
      svbool_t pg = svwhilelt_b64(j, end);
      svst1_f64(pg, &b[j], svmul_f64_x(pg, s, svld1_f64(pg, &c[j])));
      j += svcntd();
    }
  }
}

static void kernel_add(void) {
  ssize_t base, j, end;
  const ssize_t block = sve_block_elems();

#pragma omp parallel for private(j, end) schedule(static)
  for (base = 0; base < STREAM_ARRAY_SIZE; base += block) {
    end = MIN(base + block, STREAM_ARRAY_SIZE);
    for (j = base; j < end;) {
      svbool_t pg = svwhilelt_b64(j, end);
      svfloat64_t va = svld1_f64(pg, &a[j]);
      svfloat64_t vb = svld1_f64(pg, &b[j]);
      svst1_f64(pg, &c[j], svadd_f64_x(pg, va, vb));
      j += svcntd();
    }
  }
}

static void kernel_triad(STREAM_TYPE scalar) {
  ssize_t base, j, end;
  const ssize_t block = sve_block_elems();
  const svfloat64_t s = svdup_f64(scalar);

#pragma omp parallel for private(j, end) schedule(static)
  for (base = 0; base < STREAM_ARRAY_SIZE; base += block) {
    end = MIN(base + block, STREAM_ARRAY_SIZE);
    for (j = base; j < end;) {
      svbool_t pg = svwhilelt_b64(j, end);
      svfloat64_t vb = svld1_f64(pg, &b[j]);
      svfloat64_t vc = svld1_f64(pg, &c[j]);
      svst1_f64(pg, &a[j], svmla_f64_x(pg, vb, s, vc));
      j += svcntd();
    }
  }
}
#else
static void kernel_write(STREAM_TYPE value) {
  ssize_t j;
#pragma omp parallel for
  for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
    a[j] = value;
  }
}

static STREAM_TYPE kernel_read(void) {
  ssize_t j;
  STREAM_TYPE total = 0.0;
#pragma omp parallel for reduction(+ : total)
  for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
    total += a[j];
  }
  return total;
}

static void kernel_copy(void) {
  ssize_t j;
#pragma omp parallel for
  for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
    c[j] = a[j];
  }
}

static void kernel_scale(STREAM_TYPE scalar) {
  ssize_t j;
#pragma omp parallel for
  for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
    b[j] = scalar * c[j];
  }
}

static void kernel_add(void) {
  ssize_t j;
#pragma omp parallel for
  for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
    c[j] = a[j] + b[j];
  }
}

static void kernel_triad(STREAM_TYPE scalar) {
  ssize_t j;
#pragma omp parallel for
  for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
    a[j] = b[j] + scalar * c[j];
  }
}
#endif

static void check_results(int enabled[NUM_KERNELS]) {
  STREAM_TYPE aj = 1.0;
  STREAM_TYPE bj = 2.0;
  STREAM_TYPE cj = 0.0;
  STREAM_TYPE scalar = 3.0;
  STREAM_TYPE a_sum_err = 0.0, b_sum_err = 0.0, c_sum_err = 0.0;
  STREAM_TYPE a_avg_err, b_avg_err, c_avg_err;
  double epsilon = sizeof(STREAM_TYPE) == 4 ? 1.e-6 : 1.e-13;
  ssize_t j;
  int k, err = 0;

  aj = 2.0E0 * aj;

  for (k = 0; k < NTIMES; k++) {
    if (enabled[K_WRITE]) {
      aj = (STREAM_TYPE)(k + 1);
    }
    if (enabled[K_COPY]) {
      cj = aj;
    }
    if (enabled[K_SCALE]) {
      bj = scalar * cj;
    }
    if (enabled[K_ADD]) {
      cj = aj + bj;
    }
    if (enabled[K_TRIAD]) {
      aj = bj + scalar * cj;
    }
  }

  for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
    a_sum_err += ABS(a[j] - aj);
    b_sum_err += ABS(b[j] - bj);
    c_sum_err += ABS(c[j] - cj);
  }

  a_avg_err = a_sum_err / (STREAM_TYPE)STREAM_ARRAY_SIZE;
  b_avg_err = b_sum_err / (STREAM_TYPE)STREAM_ARRAY_SIZE;
  c_avg_err = c_sum_err / (STREAM_TYPE)STREAM_ARRAY_SIZE;

  if (ABS(a_avg_err / aj) > epsilon) {
    err++;
    printf("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",
           epsilon);
  }
  if (ABS(b_avg_err / bj) > epsilon) {
    err++;
    printf("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",
           epsilon);
  }
  if (ABS(c_avg_err / cj) > epsilon) {
    err++;
    printf("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",
           epsilon);
  }

  if (err == 0) {
    printf("Solution Validates: avg error less than %e on all three arrays\n",
           epsilon);
  }
}

int main(int argc, char **argv) {
  int enabled[NUM_KERNELS] = {0};
  double avgtime[NUM_KERNELS] = {0.0};
  double maxtime[NUM_KERNELS] = {0.0};
  double mintime[NUM_KERNELS] = {DBL_MAX, DBL_MAX, DBL_MAX,
                                 DBL_MAX, DBL_MAX, DBL_MAX};
  double times[NUM_KERNELS][NTIMES];
  int bytes_per_word = sizeof(STREAM_TYPE);
  int quantum, k;
  ssize_t j;
  STREAM_TYPE scalar = 3.0;

  printf(HLINE);
  printf("STREAM ARM SVE variant\n");
  if (parse_mode(argc, argv, enabled) < 0) {
    printf("Usage: %s [all|wr_rd|write|read|copy|scale|add|triad]\n", argv[0]);
    return 1;
  }
  printf("SVE intrinsics: %s\n", HAVE_SVE ? "enabled" : "not enabled, scalar fallback");
#if HAVE_SVE
  printf("SVE vector length: %zu bits\n", svcntb() * 8);
#endif
  printf("Selected kernels:");
  for (j = 0; j < NUM_KERNELS; j++) {
    if (enabled[j]) {
      printf(" %s", labels[j]);
    }
  }
  printf("\n");
  printf(HLINE);

  printf("This system uses %d bytes per array element.\n", bytes_per_word);
  printf("Array size = %llu (elements), Offset = %d (elements)\n",
         (unsigned long long)STREAM_ARRAY_SIZE, OFFSET);
  printf("Memory per array = %.1f MiB (= %.1f GiB).\n",
         bytes_per_word * ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0),
         bytes_per_word *
             ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0 / 1024.0));
  printf("Total memory required = %.1f MiB (= %.1f GiB).\n",
         (3.0 * bytes_per_word) *
             ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0),
         (3.0 * bytes_per_word) *
             ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0 / 1024.0));
  printf("Each kernel will be executed %d times.\n", NTIMES);
  printf(" The *best* time for each kernel (excluding the first iteration)\n");
  printf(" will be used to compute the reported bandwidth.\n");

#ifdef _OPENMP
  printf(HLINE);
#pragma omp parallel
  {
#pragma omp master
    printf("Number of Threads requested = %d\n", omp_get_num_threads());
  }
#endif

#pragma omp parallel for
  for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }

  printf(HLINE);
  quantum = checktick();
  if (quantum >= 1) {
    printf("Your clock granularity/precision appears to be %d microseconds.\n",
           quantum);
  } else {
    printf("Your clock granularity appears to be less than one microsecond.\n");
    quantum = 1;
  }
  printf(HLINE);

  kernel_write(2.0);

  for (k = 0; k < NTIMES; k++) {
    STREAM_TYPE write_value = (STREAM_TYPE)(k + 1);
    STREAM_TYPE read_sum;

    if (enabled[K_WRITE]) {
      times[K_WRITE][k] = mysecond();
      kernel_write(write_value);
      times[K_WRITE][k] = mysecond() - times[K_WRITE][k];
    } else {
      times[K_WRITE][k] = 0.0;
    }

    if (enabled[K_READ]) {
      times[K_READ][k] = mysecond();
      read_sum = kernel_read();
      times[K_READ][k] = mysecond() - times[K_READ][k];
      read_sink = read_sum;
    } else {
      times[K_READ][k] = 0.0;
    }

    if (enabled[K_COPY]) {
      times[K_COPY][k] = mysecond();
      kernel_copy();
      times[K_COPY][k] = mysecond() - times[K_COPY][k];
    } else {
      times[K_COPY][k] = 0.0;
    }

    if (enabled[K_SCALE]) {
      times[K_SCALE][k] = mysecond();
      kernel_scale(scalar);
      times[K_SCALE][k] = mysecond() - times[K_SCALE][k];
    } else {
      times[K_SCALE][k] = 0.0;
    }

    if (enabled[K_ADD]) {
      times[K_ADD][k] = mysecond();
      kernel_add();
      times[K_ADD][k] = mysecond() - times[K_ADD][k];
    } else {
      times[K_ADD][k] = 0.0;
    }

    if (enabled[K_TRIAD]) {
      times[K_TRIAD][k] = mysecond();
      kernel_triad(scalar);
      times[K_TRIAD][k] = mysecond() - times[K_TRIAD][k];
    } else {
      times[K_TRIAD][k] = 0.0;
    }
  }

  for (k = 1; k < NTIMES; k++) {
    for (j = 0; j < NUM_KERNELS; j++) {
      if (!enabled[j]) {
        continue;
      }
      avgtime[j] += times[j][k];
      mintime[j] = MIN(mintime[j], times[j][k]);
      maxtime[j] = MAX(maxtime[j], times[j][k]);
    }
  }

  printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
  for (j = 0; j < NUM_KERNELS; j++) {
    if (!enabled[j]) {
      continue;
    }
    avgtime[j] = avgtime[j] / (double)(NTIMES - 1);
    printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", labels[j],
           1.0E-06 * bytes[j] / mintime[j], avgtime[j], mintime[j],
           maxtime[j]);
  }
  printf(HLINE);

  check_results(enabled);
  printf(HLINE);

  return 0;
}
