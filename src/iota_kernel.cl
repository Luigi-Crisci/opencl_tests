__kernel void iota(__global float* a,
                       const unsigned int n)
{
    int id = get_global_id(0);
    if (id < n)
        a[id] = n;
}