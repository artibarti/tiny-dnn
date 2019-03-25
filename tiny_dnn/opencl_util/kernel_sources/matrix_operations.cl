
__kernel void matrixMul(const int BLOCK_SIZE, __global float* left, __global float* right,
    __global float* result, const int left_h, const int left_w, const int right_h, const int right_w)
{
    int global_x = get_group_id(0);
    int global_y = get_group_id(1);
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int row_index = global_y * BLOCK_SIZE + local_y;
    int col_index = global_x * BLOCK_SIZE + local_x;

    if (row_index < left_h && col_index < right_w) {

        float sum = 0;
        for (int i = 0; i<left_w; i++) {
            sum += left[row_index * left_w + i] * right[i * right_w + col_index];
        }
        result[row_index * right_w + col_index] = sum;
    }
}