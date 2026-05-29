/*-----------------------------------------------------------------------*/
/* Program: STREAM all-kernel variant                                    */
/*                                                                       */
/* Based on STREAM 5.10 by John D. McCalpin and Joe R. Zagar.            */
/* This variant appends Write and Read kernels to the regular STREAM      */
/* Copy, Scale, Add, and Triad kernels.                                  */
/*-----------------------------------------------------------------------*/
# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
# include <string.h>

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif

#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif

#ifndef OFFSET
#   define OFFSET	0
#endif

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

#define NUM_KERNELS 6
#define K_WRITE 0
#define K_READ 1
#define K_COPY 2
#define K_SCALE 3
#define K_ADD 4
#define K_TRIAD 5

static STREAM_TYPE	a[STREAM_ARRAY_SIZE+OFFSET],
			b[STREAM_ARRAY_SIZE+OFFSET],
			c[STREAM_ARRAY_SIZE+OFFSET];

static double	avgtime[NUM_KERNELS] = {0}, maxtime[NUM_KERNELS] = {0},
		mintime[NUM_KERNELS] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[NUM_KERNELS] = {
    "Write:     ", "Read:      ", "Copy:      ",
    "Scale:     ", "Add:       ", "Triad:     "
    };

static double	bytes[NUM_KERNELS] = {
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE
    };

static volatile STREAM_TYPE read_sink;

extern double mysecond();
extern void checkSTREAMresults(int enabled[NUM_KERNELS]);
extern int parse_mode(int argc, char **argv, int enabled[NUM_KERNELS]);
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif
int
main(int argc, char **argv)
    {
    int			quantum, checktick();
    int			BytesPerWord;
    int			k;
    int			enabled[NUM_KERNELS] = {0};
    ssize_t		j;
    STREAM_TYPE		scalar;
    STREAM_TYPE		read_sum;
    STREAM_TYPE		write_value;
    double		t, times[NUM_KERNELS][NTIMES];

    printf(HLINE);
    printf("STREAM all-kernel variant\n");
    if (parse_mode(argc, argv, enabled) < 0) {
	printf("Usage: %s [all|wr_rd|write|read|copy|scale|add|triad]\n", argv[0]);
	return 1;
    }
    printf("Selected kernels:");
    for (j=0; j<NUM_KERNELS; j++)
	if (enabled[j])
	    printf(" %s", label[j]);
    printf("\n");
    printf(HLINE);
    BytesPerWord = sizeof(STREAM_TYPE);
    printf("This system uses %d bytes per array element.\n",
	BytesPerWord);

    printf(HLINE);
    printf("Array size = %llu (elements), Offset = %d (elements)\n" , (unsigned long long) STREAM_ARRAY_SIZE, OFFSET);
    printf("Memory per array = %.1f MiB (= %.1f GiB).\n",
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0),
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0/1024.0));
    printf("Total memory required = %.1f MiB (= %.1f GiB).\n",
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.),
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024./1024.));
    printf("Each kernel will be executed %d times.\n", NTIMES);
    printf(" The *best* time for each kernel (excluding the first iteration)\n");
    printf(" will be used to compute the reported bandwidth.\n");

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel
    {
#pragma omp master
	{
	    k = omp_get_num_threads();
	    printf ("Number of Threads requested = %i\n",k);
        }
    }
#endif

#ifdef _OPENMP
	k = 0;
#pragma omp parallel
#pragma omp atomic
		k++;
    printf ("Number of Threads counted = %i\n",k);
#endif

#pragma omp parallel for
    for (j=0; j<STREAM_ARRAY_SIZE; j++) {
	    a[j] = 1.0;
	    b[j] = 2.0;
	    c[j] = 0.0;
	}

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1)
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
    else {
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");
	quantum = 1;
    }

    t = mysecond();
#pragma omp parallel for
    for (j = 0; j < STREAM_ARRAY_SIZE; j++)
		a[j] = 2.0E0 * a[j];
    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);

    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
	write_value = (STREAM_TYPE) (k + 1);

	if (enabled[K_WRITE]) {
	times[K_WRITE][k] = mysecond();
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    a[j] = write_value;
	times[K_WRITE][k] = mysecond() - times[K_WRITE][k];
	} else {
	times[K_WRITE][k] = 0.0;
	}

	if (enabled[K_READ]) {
	read_sum = 0.0;
	times[K_READ][k] = mysecond();
#pragma omp parallel for reduction(+:read_sum)
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    read_sum += a[j];
	times[K_READ][k] = mysecond() - times[K_READ][k];
	read_sink = read_sum;
	} else {
	times[K_READ][k] = 0.0;
	}

	if (enabled[K_COPY]) {
	times[K_COPY][k] = mysecond();
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    c[j] = a[j];
	times[K_COPY][k] = mysecond() - times[K_COPY][k];
	} else {
	times[K_COPY][k] = 0.0;
	}

	if (enabled[K_SCALE]) {
	times[K_SCALE][k] = mysecond();
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    b[j] = scalar*c[j];
	times[K_SCALE][k] = mysecond() - times[K_SCALE][k];
	} else {
	times[K_SCALE][k] = 0.0;
	}

	if (enabled[K_ADD]) {
	times[K_ADD][k] = mysecond();
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    c[j] = a[j]+b[j];
	times[K_ADD][k] = mysecond() - times[K_ADD][k];
	} else {
	times[K_ADD][k] = 0.0;
	}

	if (enabled[K_TRIAD]) {
	times[K_TRIAD][k] = mysecond();
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    a[j] = b[j]+scalar*c[j];
	times[K_TRIAD][k] = mysecond() - times[K_TRIAD][k];
	} else {
	times[K_TRIAD][k] = 0.0;
	}
	}

    for (k=1; k<NTIMES; k++)
	{
	for (j=0; j<NUM_KERNELS; j++)
	    {
	    if (!enabled[j])
		continue;
	    avgtime[j] = avgtime[j] + times[j][k];
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}

    printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
    for (j=0; j<NUM_KERNELS; j++) {
		if (!enabled[j])
		    continue;
		avgtime[j] = avgtime[j]/(double)(NTIMES-1);

		printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       avgtime[j],
	       mintime[j],
	       maxtime[j]);
    }
    printf(HLINE);

    checkSTREAMresults(enabled);
    printf(HLINE);

    return 0;
}

int
parse_mode(int argc, char **argv, int enabled[NUM_KERNELS])
    {
    int i;
    if (argc <= 1 || strcmp(argv[1], "all") == 0) {
	for (i=0; i<NUM_KERNELS; i++)
	    enabled[i] = 1;
	return 0;
    }
    if (strcmp(argv[1], "wr_rd") == 0 || strcmp(argv[1], "read_write") == 0) {
	enabled[K_WRITE] = 1;
	enabled[K_READ] = 1;
	return 0;
    }
    if (strcmp(argv[1], "write") == 0)
	enabled[K_WRITE] = 1;
    else if (strcmp(argv[1], "read") == 0)
	enabled[K_READ] = 1;
    else if (strcmp(argv[1], "copy") == 0)
	enabled[K_COPY] = 1;
    else if (strcmp(argv[1], "scale") == 0)
	enabled[K_SCALE] = 1;
    else if (strcmp(argv[1], "add") == 0)
	enabled[K_ADD] = 1;
    else if (strcmp(argv[1], "triad") == 0)
	enabled[K_TRIAD] = 1;
    else
	return -1;
    return 0;
    }

# define	M	20

int
checktick()
    {
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

    for (i = 0; i < M; i++) {
	t1 = mysecond();
	while( ((t2=mysecond()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
    }

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
void checkSTREAMresults (int enabled[NUM_KERNELS])
{
	STREAM_TYPE aj,bj,cj,scalar;
	STREAM_TYPE aSumErr,bSumErr,cSumErr;
	STREAM_TYPE aAvgErr,bAvgErr,cAvgErr;
	double epsilon;
	ssize_t	j;
	int	k,err;

	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
	aj = 2.0E0 * aj;
	scalar = 3.0;

	for (k=0; k<NTIMES; k++) {
		if (enabled[K_WRITE])
			aj = (STREAM_TYPE) (k + 1);
		if (enabled[K_COPY])
			cj = aj;
		if (enabled[K_SCALE])
			bj = scalar*cj;
		if (enabled[K_ADD])
			cj = aj+bj;
		if (enabled[K_TRIAD])
			aj = bj+scalar*cj;
	}

	aSumErr = 0.0;
	bSumErr = 0.0;
	cSumErr = 0.0;
	for (j=0; j<STREAM_ARRAY_SIZE; j++) {
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);
	}
	aAvgErr = aSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	bAvgErr = bSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	cAvgErr = cSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;

	if (sizeof(STREAM_TYPE) == 4) {
		epsilon = 1.e-6;
	}
	else if (sizeof(STREAM_TYPE) == 8) {
		epsilon = 1.e-13;
	}
	else {
		printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n",sizeof(STREAM_TYPE));
		epsilon = 1.e-6;
	}

	err = 0;
	if (abs(aAvgErr/aj) > epsilon) {
		err++;
		printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
	}
	if (abs(bAvgErr/bj) > epsilon) {
		err++;
		printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
	}
	if (abs(cAvgErr/cj) > epsilon) {
		err++;
		printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
	}
	if (err == 0) {
		printf ("Solution Validates: avg error less than %e on all three arrays\n",epsilon);
	}
}
