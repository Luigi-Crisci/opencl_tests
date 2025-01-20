template<class T>
T add( T x, T y )
{
    return x + y;
}

__kernel void test(__global float* a, __global float* b)
{
    uint index = get_global_id(0);

    float x = b[ index ];
    float y = b[ index + 1 ];

    a[ index ] = add( x, y );
}

