#include <cstdlib>
#include <iostream>
#include <thrust/sort.h>
#include <set>
#include <queue>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#define BLOCK_SIZE 512 // number of threads per block

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;

typedef struct point_desc
{
    double x_pos;
    double y_pos;
} point;

typedef struct line_desc
{
    point start_point;
    point end_point;
} line;

typedef struct vector_desc
{
    double x_pos;
    double y_pos;
} vector;

// These are the functions used

__host__ __device__ bool comp_point(const point p1, const point p2)
{
    if (p1.y_pos == p2.y_pos)
        return p1.x_pos < p2.x_pos;
    return p1.y_pos < p2.y_pos;
}

__host__ __device__ bool comp_line_y_start(const line l1, const line l2)
{
    return l1.start_point.y_pos < l2.start_point.y_pos;
}

__host__ __device__ double cross(vector l0, vector l1)
{
    return l0.x_pos*l1.y_pos - l0.y_pos*l1.x_pos;
}

__host__ __device__ bool intersect_test(line l0, line l1)
{    
    //Given two lines, we have to evaluate whether the start/end points are on "different sides" of the line, which only happens if their cross products change in sign.
    //So, we need to establish vectors to test with.
    vector l01, l03, l04;
    l01.x_pos = l0.end_point.x_pos-l0.start_point.x_pos;
    l01.y_pos = l0.end_point.y_pos-l0.start_point.y_pos;
    l03.x_pos = l1.start_point.x_pos-l0.start_point.x_pos;
    l03.y_pos = l1.start_point.y_pos-l0.start_point.y_pos;
    l04.x_pos = l1.end_point.x_pos-l0.start_point.x_pos;
    l04.y_pos = l1.end_point.y_pos-l0.start_point.y_pos;
    
    double z1 = cross(l01, l03);
    double z2 = cross(l01, l04);
    //These need to differ in orientation, but also allow for points to lie on another line- that also counts as an intersection
    if(z1 < 0 && z2 < 0)
        return false;
    if(z1 > 0 && z2 > 0)
        return false;
    if(z1 == 0 && z2 == 0)
        return false;
    
    //Repeat the same process, but with other as the base
    vector l34, l30, l31;
    l34.x_pos = l1.start_point.x_pos-l1.end_point.x_pos;
    l34.y_pos = l1.start_point.y_pos-l1.end_point.y_pos;
    l30.x_pos = l0.start_point.x_pos-l1.start_point.x_pos;
    l30.y_pos = l0.start_point.y_pos-l1.start_point.y_pos;
    l31.x_pos = l0.end_point.x_pos-l1.start_point.x_pos;
    l31.y_pos = l0.end_point.y_pos-l1.start_point.y_pos;
    
    double z3 = cross(l34,l30);
    double z4 = cross(l34,l31);
    //These need to differ in orientation, but also allow for points to lie on another line- that also counts as an intersection
    if(z3 < 0 && z4 < 0)
        return false;
    if(z3 > 0 && z4 > 0)
        return false;
    if(z3 == 0 && z4 == 0)
        return false;
    
    //If in both cases the orientations aren't the same, then they intersect.
    return true;
}
   
__host__ __device__ point intersection_point(line l0, line l1){
  //Equation for parametric form is P+Dt
  double x1 = l0.start_point.x_pos;
  double y1 = l0.start_point.y_pos;
  double x2 = l1.start_point.x_pos;
  double y2 = l1.start_point.y_pos;
  double Dx1 = l0.end_point.x_pos-l0.start_point.x_pos;
  double Dy1 = l0.end_point.y_pos-l0.start_point.y_pos;
  double Dx2 = l1.end_point.x_pos-l1.start_point.x_pos;
  double Dy2 = l1.end_point.y_pos-l1.start_point.y_pos;
  
  //Given an equation start_pointx+dx1t1 = end_pointx+dx2t2, t2=(start_pointx-end_pointx+dx1t1)/dx2, start_pointy+dy1t1 = end_pointy+dy2t2, t1 = (end_pointy-start_pointy+dy2t2)/dy1 = (end_pointy-start_pointy+dy2(start_pointx-end_pointx+dx1t1)/dx2)/dy1; switch x's and y's if any are zero. If zero in both cases then they do not overlap and should not have ended up here anyway
  //Note: botched the numbers a bit above, actual equation (thank you automatic mathematics solving in Word) is  t2 = (-D_y1  (x_1-x_2 )+y_1  D_x1-y_2  D_x1)/(D_x2  D_y1-D_x1  D_y2 ) and t1 = -(-D_y2  (x_1-x_2 )+y_1  D_x2-y_2  D_x2)/(D_x2  D_y1-D_x1  D_y2 )
  //Fortunately for us, the denominator has exactly one instance where it is zero- if Dx2Dy1 = Dx1Dy2.
  if(Dx2*Dy1 != Dx1*Dy2)
  {
    float t1 = -(-Dy1*(x1-x2)+y1*Dx1-y2*Dx1)/(Dx2*Dy1-Dx1*Dy2);
    float t2 = -(-Dy2*(x1-x2)+y1*Dx2-y2*Dx2)/(Dx2*Dy1-Dx1*Dy2);
    if (t1 < 0 || t1 > 1 || t2 < 0 || t2 > 1)
    {
        point q;
        q.x_pos = 0;
        q.y_pos = 0;
        return q;
    }
    point p;
    p.x_pos = l0.start_point.x_pos+Dx1*t2;
    p.y_pos = l0.start_point.y_pos+Dy1*t2;
    return p;   
  }
  else
  {
      point q;
      q.x_pos = 0;
      q.y_pos = 0;
      return q;
  }
}

point* trim(point* points, int& len)
{
    int i = 0;
    while(i < len && points[i].x_pos == 0 && points[i].y_pos == 0)
        i++;
    if(i == len)
        return points;
    point* temp = (point*)malloc(sizeof(point)*(len-i));
    memcpy(temp, points+i, sizeof(point)*(len-i));
    len -= i;
    return temp;
}

struct event
{
  bool state;
  double value;
  line e;
};

struct comp_event
{
    bool operator()(event e1, event e2) const
    {
        if(e1.value == e2.value)
        {
            if(e1.state && !e2.state)
                return false;
            return true;
        }
        return e1.value < e2.value;
    }
};

struct comp_line
{
    bool operator()(line l1, line l2) const
    {
        if(l1.start_point.y_pos == l2.start_point.y_pos)
        {
            if(l1.start_point.x_pos == l2.start_point.x_pos)
            {
                if(l1.end_point.y_pos == l2.end_point.y_pos)
                {
                    return l1.end_point.x_pos < l2.end_point.x_pos;
                }
                return l1.end_point.y_pos < l2.end_point.y_pos;
            }
            return l1.start_point.x_pos < l2.start_point.x_pos;
        }
        return l1.start_point.y_pos < l2.start_point.y_pos;
    }
};

point* line_intersect_baseline(line* lines, int len)
{
    point* intersection_points = (point *) malloc(len*len*sizeof(point));
    for(int i = 0; i < len; ++i)
        for(int j = i+1; j < len; ++j)
            if (intersect_test(lines[i],lines[j]))
                intersection_points[i*len + j] = intersection_point(lines[i],lines[j]);

    return intersection_points;
}

bool in_interval (line p1, line p2)
{
  double x1left = std::min(p1.start_point.x_pos, p1.end_point.x_pos);
  double x1right = std::max(p1.start_point.x_pos, p1.end_point.x_pos);
  double x2left = std::min(p2.start_point.x_pos, p2.end_point.x_pos);
  double x2right = std::max(p2.start_point.x_pos, p2.end_point.x_pos);
  return !(x2right < x1left || x2left > x1right);
}

point* line_intersect_optimal(line* lines, int& len)
{
    //To do the "line" method with axis-aligned bounding boxes, we need to simulate that line. We do that with a priority queue of every time we either add or remove a box, given their y-values and the edge that they refer to
    std::vector<point> output;
    std::priority_queue<event, std::vector<event>, comp_event> queue;
    for(int i = 0; i < len; ++i)
    {
      if(lines[i].start_point.y_pos <= lines[i].end_point.y_pos)
      {
        event e1;
        e1.state = true;
        e1.value = lines[i].start_point.y_pos;
        e1.e = lines[i];
        event e2;
        e2.state = false;
        e2.value = lines[i].end_point.y_pos;
        e2.e = lines[i];
        queue.push(e1);
        queue.push(e2);
      }
      else
      {
        event e1;
        e1.state = false;
        e1.value = lines[i].start_point.y_pos;
        e1.e = lines[i];
        event e2;
        e2.state = true;
        e2.value = lines[i].end_point.y_pos;
        e2.e = lines[i];
        queue.push(e1);
        queue.push(e2);
      }
    }
    //Now we need our "active set"
    std::set<line,comp_line> active;
    //In this implementation we will be lazy and just compare against everything in the active set, as opposed to establishing an AVL tree
    while(queue.size()>0)
    {
      event e = queue.top();
      queue.pop();
      if(!e.state)
        active.erase(e.e);
      else
      {
        for(std::set<line,comp_line>::iterator it = active.begin(); it != active.end(); ++it)
          if (in_interval(e.e, *it) && intersect_test(e.e, *it))
            output.push_back(intersection_point(e.e, *it));
        active.insert(e.e);
      }
    }
    int output_len = output.size();
    point* output_array = (point*)malloc(sizeof(point)*output_len);
    for(int i = 0; i < output_len; ++i)
        output_array[i] = output[i];
    len = output_len;
    return output_array;
}

__global__ void line_intersect_brute_unoptimized_device(line* d_lines, point* d_intersection_points, int d_len) // GPU kernel for unoptimized brute-force method
{
    int i = threadIdx.x + blockDim.x * blockIdx.x, j; // assign to each thread their respective location in the overall array of threads- think like turrning a 2-dimensional array into a 1-dimensional array
    if (i < d_len)
    {
        line base_line = d_lines[i];
		for(j = i+1; j < d_len; j++) // Simple brute force- compare line against every other line
            if(intersect_test(base_line,d_lines[j]))
                d_intersection_points[i*d_len + j] = intersection_point(base_line,d_lines[j]);
    }
}

__global__ void line_intersect_brute_optimized_device(line* d_lines, point* d_intersection_points, int d_len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x, j, k, offset, c;
	extern __shared__ line s_base_array[]; // array we use to store a local copy of a segment of the line segments in local memory to avoid overhead in memory calls
	line* s_line_list = s_base_array;
	line local_entry;
	if (i < d_len)
		local_entry = d_lines[i]; // grab, from our global array, the local value that we will be checking against with this thread as we check against all tiles other than this one
    // Offset logic- think about flattening a trinagle made of squares into a rectangle. Depending on the number of squares, you can either get a proper rectangle, or a rectangle where half has one more block on top than the other
    offset = 1;
	if(gridDim.x % 2 == 0)
		if(blockIdx.x >= gridDim.x/2)
			offset = 0;
	for(j = (blockIdx.x+1)%gridDim.x, c = 1; c < gridDim.x/2 + offset; j=(j+1)%gridDim.x, ++c) // iterate through all of our tiles
	{
        // copy tile to local memory
		if(threadIdx.x+j*blockDim.x < d_len)
            s_line_list[threadIdx.x] = d_lines[threadIdx.x+j*blockDim.x];
        __syncthreads();
        // brute force against tile
		for(k = 0; k < blockDim.x; ++k)
			if(k+j*blockDim.x < d_len && i < d_len)
                if(intersect_test(local_entry,s_line_list[k]))
                    d_intersection_points[i*d_len + k + j * blockDim.x] = intersection_point(local_entry,s_line_list[k]);
		__syncthreads();
    }
    // create final tile- the tile that overlaps with the data these threads are processing
	if (i < d_len)
        s_line_list[threadIdx.x] = local_entry;
	__syncthreads();
    
    // brute-force over this
	for(j = threadIdx.x+1; j < blockDim.x; ++j)
		if(j + blockIdx.x * blockDim.x < d_len)
            if(intersect_test(local_entry,s_line_list[j]))
                d_intersection_points[i*d_len + j + blockIdx.x * blockDim.x] = intersection_point(local_entry,s_line_list[j]);
}

__host__ float line_intersect_brute_unoptimized_host(line* lines, point* intersection_points, int len) // host kernel for GPU implementation of unoptimized brute-force method
{
	size_t size_input = len * sizeof(line);
	size_t size_output = len * len * sizeof(point);
	line * d_lines;
    point * d_intersection_points;

	cudaMalloc((void**) &d_lines, size_input);
	cudaMemcpy(d_lines, lines, size_input, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_intersection_points, size_output);
	cudaMemset(d_intersection_points, 0, size_output);

	dim3 DimGrid(len/BLOCK_SIZE, 1, 1);
	if (len % BLOCK_SIZE) DimGrid.x++;
	dim3 DimBlock(BLOCK_SIZE, 1, 1);

	/* Run time measurement start */
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	line_intersect_brute_unoptimized_device<<<DimGrid,DimBlock>>>(d_lines, d_intersection_points, len);
	cudaEventRecord(stop,0);

	cudaMemcpy(intersection_points, d_intersection_points, size_output, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Running time for unoptimized version: %0.5f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_lines);
	cudaFree(d_intersection_points);

	return elapsedTime;
}


__host__ float line_intersect_brute_optimized_host(line* lines, point* intersection_points, int len) // host kernel for GPU implementation of optimized brute-force method
{
	size_t size_input = len * sizeof(line);
	size_t size_output = len * len * sizeof(point);
	line * d_lines;
    point * d_intersection_points;

	cudaMalloc((void**) &d_lines, size_input);
	cudaMemcpy(d_lines, lines, size_input, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_intersection_points, size_output);
	cudaMemset(d_intersection_points, 0, size_output);

	dim3 DimGrid(len/BLOCK_SIZE, 1, 1);
	if (len % BLOCK_SIZE) DimGrid.x++;
	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	size_t shared_memory_tile_size = sizeof(line)*BLOCK_SIZE;
	size_t shared_memory_size = shared_memory_tile_size; // We use this to load, locally, a block of components into the GPU to avoid constantly pulling in stuff from our global memory- d_lines

	/* Run time measurement start */
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	line_intersect_brute_optimized_device<<<DimGrid,DimBlock,shared_memory_size>>>(d_lines, d_intersection_points, len);
	cudaEventRecord(stop,0);

	cudaMemcpy(intersection_points, d_intersection_points, size_output, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Running time for optimized version: %0.5f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_lines);
	cudaFree(d_intersection_points);

	return elapsedTime;
}

void populate(line* line_array, int len, int range) // Fills array with random start and end points in the range, composed as lines
{
	std::srand(1);
    for(int i = 0; i < len; ++i)
    {
        line l;
        point s,e;
        s.x_pos = std::rand() % range;
        s.y_pos = std::rand() % range;
        e.x_pos = std::rand() % range;
        e.y_pos = std::rand() % range;
        l.start_point = s;
        l.end_point = e;
        line_array[i] = l;
    }
}




/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

int main(int argc, char** argv)
{
    int line_count = std::atoi(argv[1]);
    int range = std::atoi(argv[2]);
    line* lines = (line*)malloc(sizeof(line)*line_count);
    populate(lines,line_count,range); // populates line array with random lines occuping the space defined on the specified interval
    int base_op_len = line_count;
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
    point* intersection_points_base = line_intersect_baseline(lines, line_count);
	/* check the total running time */ 
	report_running_time();
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
    point* base_op_temp = line_intersect_optimal(lines, base_op_len);
	/* check the total running time */ 
	report_running_time();

    point * intersection_points_brute_unoptimized = (point *)malloc(sizeof(point)*line_count*line_count);
    memset(intersection_points_brute_unoptimized,0,sizeof(point)*line_count*line_count);
    point * intersection_points_brute_optimized = (point *)malloc(sizeof(point)*line_count*line_count);
    memset(intersection_points_brute_unoptimized,0,sizeof(point)*line_count*line_count);

    line_intersect_brute_unoptimized_host(lines, intersection_points_brute_unoptimized, line_count);
    line_intersect_brute_optimized_host(lines, intersection_points_brute_optimized, line_count);
    /* Disabling error-checking
    thrust::sort(intersection_points_base, intersection_points_base + line_count*line_count, comp_point);
    std::sort(base_op_temp, base_op_temp + base_op_len, comp_point);
    thrust::sort(intersection_points_brute_unoptimized, intersection_points_brute_unoptimized + line_count*line_count, comp_point);
    thrust::sort(intersection_points_brute_optimized, intersection_points_brute_optimized + line_count*line_count, comp_point);
    int len1 = line_count * line_count;
    int len2 = line_count * line_count;
    int len3 = line_count * line_count;
    int len4 = base_op_len;
    point* base_unop = trim(intersection_points_base, len1);
    point* b_unop = trim(intersection_points_brute_unoptimized, len2);
    point* b_op = trim(intersection_points_brute_optimized, len3);
    point* base_op = trim(base_op_temp, len4);
    int error = 0;
    for(int i = 0; i < std::min(len1, len2); ++i)
        error += base_unop[i].x_pos != b_unop[i].x_pos || base_unop[i].y_pos != b_unop[i].y_pos;
    std::cout << error << std::endl;
    error = 0;
    for(int i = 0; i < std::min(len1, len3); ++i)
        error += base_unop[i].x_pos != b_op[i].x_pos || base_unop[i].y_pos != b_op[i].y_pos;
    std::cout << error << std::endl;
    error = 0;
    for(int i = 0; i < std::min(len1, base_op_len); ++i)
        error += base_unop[i].x_pos != base_op[i].x_pos || base_unop[i].y_pos != base_op[i].y_pos;
    std::cout << error << std::endl;
    */

    /*
    free(base_unop);
    free(base_op);
    free(b_unop);
    free(b_op);
    */
    free(base_op_temp);
    free(lines);
    free(intersection_points_base);
    free(intersection_points_brute_unoptimized);
    free(intersection_points_brute_optimized);
    return 0;
}