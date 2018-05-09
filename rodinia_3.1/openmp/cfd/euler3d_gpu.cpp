// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>

#ifdef OMP_GPU_OFFLOAD_UM
//#define CUDA_UM
#define MAP_ALL
#include <cuda_runtime_api.h>
#endif

#if defined(OMP_GPU_OFFLOAD) || defined(OMP_GPU_OFFLOAD_UM)
#pragma omp declare target
#endif
#include <cmath>
#if defined(OMP_GPU_OFFLOAD) || defined(OMP_GPU_OFFLOAD_UM)
#pragma omp end declare target
#endif

#include <omp.h>

struct float3d { float x, y, z; };

#ifndef block_length
#ifdef _OPENMP
#error "you need to define block_length"
#else
#define block_length 1
#endif
#endif

/*
 * Options
 *
 */
#define GAMMA 1.4
#define iterations 20

#define NDIM 3
#define NNB 4

#define RK 3    // 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

/*
 * not options
 */
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


#ifdef restrict
#define __restrict restrict
#else
#define __restrict 
#endif

/*
 * Generic functions
 */
template <typename T> T* alloc(unsigned long long N)
{
    return new T[N];
}

template <typename T> void dealloc(T* array)
{
    delete[] array;
}

template <typename T> void copy(T* dst, T* src, unsigned long long N)
{
    #ifdef OMP_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for default(shared) schedule(static)
    #elif defined(OMP_GPU_OFFLOAD_UM) && defined(MAP_ALL)
    #pragma omp target teams distribute parallel for
    #elif defined(OMP_GPU_OFFLOAD_UM)
    #pragma omp target teams distribute parallel for \
                is_device_ptr(dst), is_device_ptr(src) map(to: N)
    #else
    #pragma omp parallel for default(shared) schedule(static)
    #endif
    for(unsigned long long i = 0; i < N; i++)
    {
        dst[i] = src[i];
    }
}

void dump(float* variables, int nel, int nelr)
{
    {
        std::ofstream file("density");
        file << nel << " " << nelr << std::endl;
        for(int i = 0; i < nel; i++) file << variables[i + VAR_DENSITY*nelr] << std::endl;
    }

    {
        std::ofstream file("momentum");
        file << nel << " " << nelr << std::endl;
        for(int i = 0; i < nel; i++)
        {
            for(int j = 0; j != NDIM; j++) file << variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
            file << std::endl;
        }
    }

    {
        std::ofstream file("density_energy");
        file << nel << " " << nelr << std::endl;
        for(int i = 0; i < nel; i++) file << variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
    }

}

void initialize_variables(int nelr, float* variables, float* ff_variable)
{
    #pragma omp parallel for default(shared) schedule(static)
    for(int i = 0; i < nelr; i++)
    {
        for(int j = 0; j < NVAR; j++) 
            variables[i + j*nelr] = ff_variable[j];
    }
}

#if defined(OMP_GPU_OFFLOAD) || defined(OMP_GPU_OFFLOAD_UM)
#pragma omp declare target
#endif
inline void compute_flux_contribution(float& density, float3d& momentum, float& density_energy, float& pressure, float3d& velocity, float3d& fc_momentum_x, 
        float3d& fc_momentum_y, float3d& fc_momentum_z, float3d& fc_density_energy)
{
    fc_momentum_x.x = velocity.x*momentum.x + pressure;
    fc_momentum_x.y = velocity.x*momentum.y;
    fc_momentum_x.z = velocity.x*momentum.z;

    fc_momentum_y.x = fc_momentum_x.y;
    fc_momentum_y.y = velocity.y*momentum.y + pressure;
    fc_momentum_y.z = velocity.y*momentum.z;

    fc_momentum_z.x = fc_momentum_x.z;
    fc_momentum_z.y = fc_momentum_y.z;
    fc_momentum_z.z = velocity.z*momentum.z + pressure;

    float de_p = density_energy+pressure;
    fc_density_energy.x = velocity.x*de_p;
    fc_density_energy.y = velocity.y*de_p;
    fc_density_energy.z = velocity.z*de_p;
}

inline void compute_velocity(float& density, float3d& momentum, float3d& velocity)
{
    velocity.x = momentum.x / density;
    velocity.y = momentum.y / density;
    velocity.z = momentum.z / density;
}

inline float compute_speed_sqd(float3d& velocity)
{
    return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

inline float compute_pressure(float& density, float& density_energy, float& speed_sqd)
{
    return (float(GAMMA)-float(1.0f))*(density_energy - float(0.5f)*density*speed_sqd);
}

inline float compute_speed_of_sound(float& density, float& pressure)
{
    return std::sqrt(float(GAMMA)*pressure/density);
}

#if defined(OMP_GPU_OFFLOAD) || defined(OMP_GPU_OFFLOAD_UM) 
#pragma omp end declare target
#endif
void compute_step_factor(int nelr, float* __restrict variables, float* areas, float* __restrict step_factors)
{
    #ifdef OMP_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for default(shared) schedule(auto)
    #elif defined(OMP_GPU_OFFLOAD_UM) && defined(MAP_ALL)
    #pragma omp target teams distribute parallel for
    #elif defined(OMP_GPU_OFFLOAD_UM)
    #pragma omp target teams distribute parallel for default(shared) schedule(auto) \
                is_device_ptr(areas) \
                is_device_ptr(step_factors) \
                is_device_ptr(variables)
    #else
    #pragma omp parallel for default(shared) schedule(auto)
    #endif
    for(int blk = 0; blk < nelr/block_length; ++blk)
    {
        int b_start = blk*block_length;
        int b_end = (blk+1)*block_length > nelr ? nelr : (blk+1)*block_length;
        #if !defined(OMP_GPU_OFFLOAD) && !defined(OMP_GPU_OFFLOAD_UM)
        #pragma omp simd
        #endif
        for(unsigned long long i = b_start; i < b_end; i++)
        {
            float density = variables[i + VAR_DENSITY*nelr];

            float3d momentum;
            momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
            momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
            momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];

            float density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];
            float3d velocity;       compute_velocity(density, momentum, velocity);
            float speed_sqd      = compute_speed_sqd(velocity);
            float pressure       = compute_pressure(density, density_energy, speed_sqd);
            float speed_of_sound = compute_speed_of_sound(density, pressure);

            step_factors[i] = float(0.5f) / (std::sqrt(areas[i]) * (std::sqrt(speed_sqd) + speed_of_sound));
        }
    }
}


/*
 *
 *
 */

void compute_flux(int nelr, int* elements_surrounding_elements, float* normals, float* variables, float* fluxes, float* ff_variable, 
        float3d ff_flux_contribution_momentum_x, float3d ff_flux_contribution_momentum_y, float3d ff_flux_contribution_momentum_z, 
        float3d ff_flux_contribution_density_energy)
{
    const float smoothing_coefficient = float(0.2f);

    #if defined(OMP_GPU_OFFLOAD)
    #pragma omp target teams distribute parallel for
    #elif defined(OMP_GPU_OFFLOAD_UM) && defined(MAP_ALL)
    #pragma omp target teams distribute parallel for
    #elif defined(OMP_GPU_OFFLOAD_UM)
    #pragma omp target teams distribute parallel for default(shared) schedule(auto) \
                    map(alloc:smoothing_coefficient) \
                    is_device_ptr(elements_surrounding_elements) \
                    is_device_ptr(normals) \
                    is_device_ptr(variables) \
                    is_device_ptr(fluxes) \
                    is_device_ptr(ff_variable)
                    
    #else
    #pragma omp parallel for default(shared) schedule(auto)
    #endif
    for(int blk = 0; blk < nelr/block_length; ++blk)
    {
        int b_start = blk*block_length;
        int b_end = (blk+1)*block_length > nelr ? nelr : (blk+1)*block_length;

        #if !defined(OMP_GPU_OFFLOAD) && !defined(OMP_GPU_OFFLOAD_UM)
        #pragma omp simd
        #endif
        for(unsigned long long i = b_start; i < b_end; ++i)
        {
            float density_i = variables[i + VAR_DENSITY*nelr];
            float3d momentum_i;
            momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
            momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
            momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];

            float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

            float3d velocity_i;                              compute_velocity(density_i, momentum_i, velocity_i);
            float speed_sqd_i                          = compute_speed_sqd(velocity_i);
            float speed_i                              = std::sqrt(speed_sqd_i);
            float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
            float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
            float3d flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
            float3d flux_contribution_i_density_energy;
            compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, 
                    flux_contribution_i_momentum_z, flux_contribution_i_density_energy);

            float flux_i_density = float(0.0f);
            float3d flux_i_momentum;
            flux_i_momentum.x = float(0.0f);
            flux_i_momentum.y = float(0.0f);
            flux_i_momentum.z = float(0.0f);
            float flux_i_density_energy = float(0.0f);

            float3d velocity_nb;
            float density_nb, density_energy_nb;
            float3d momentum_nb;
            float3d flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
            float3d flux_contribution_nb_density_energy;
            float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

            #pragma unroll
            for(unsigned long long j = 0; j < NNB; j++)
            {
                float3d normal; float normal_len;
                float factor;

                long long nb = elements_surrounding_elements[i + j*nelr];
                normal.x = normals[i + (j + 0*NNB)*nelr];
                normal.y = normals[i + (j + 1*NNB)*nelr];
                normal.z = normals[i + (j + 2*NNB)*nelr];
                normal_len = std::sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);

                if(nb >= 0)     // a legitimate neighbor
                {
                    density_nb =        variables[nb + VAR_DENSITY*nelr];
                    momentum_nb.x =     variables[nb + (VAR_MOMENTUM+0)*nelr];
                    momentum_nb.y =     variables[nb + (VAR_MOMENTUM+1)*nelr];
                    momentum_nb.z =     variables[nb + (VAR_MOMENTUM+2)*nelr];
                    density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
                    compute_velocity(density_nb, momentum_nb, velocity_nb);
                    speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
                    pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
                    speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
                    compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, 
                            flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);

                    // artificial viscosity
                    factor = -normal_len*smoothing_coefficient*float(0.5f)*(speed_i + std::sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
                    flux_i_density += factor*(density_i-density_nb);
                    flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
                    flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
                    flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
                    flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

                    // accumulate cell-centered fluxes
                    factor = float(0.5f)*normal.x;
                    flux_i_density += factor*(momentum_nb.x+momentum_i.x);
                    flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x+flux_contribution_i_density_energy.x);
                    flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x+flux_contribution_i_momentum_x.x);
                    flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x+flux_contribution_i_momentum_y.x);
                    flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x+flux_contribution_i_momentum_z.x);

                    factor = float(0.5f)*normal.y;
                    flux_i_density += factor*(momentum_nb.y+momentum_i.y);
                    flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y+flux_contribution_i_density_energy.y);
                    flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y+flux_contribution_i_momentum_x.y);
                    flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y+flux_contribution_i_momentum_y.y);
                    flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y+flux_contribution_i_momentum_z.y);

                    factor = float(0.5f)*normal.z;
                    flux_i_density += factor*(momentum_nb.z+momentum_i.z);
                    flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z+flux_contribution_i_density_energy.z);
                    flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z+flux_contribution_i_momentum_x.z);
                    flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z+flux_contribution_i_momentum_y.z);
                    flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z+flux_contribution_i_momentum_z.z);
                }
                else if(nb == -1)    // a wing boundary
                {
                    flux_i_momentum.x += normal.x*pressure_i;
                    flux_i_momentum.y += normal.y*pressure_i;
                    flux_i_momentum.z += normal.z*pressure_i;
                }
                else if(nb == -2) // a far field boundary
                {
                    factor = float(0.5f)*normal.x;
                    flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
                    flux_i_density_energy += factor*(ff_flux_contribution_density_energy.x+flux_contribution_i_density_energy.x);
                    flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.x + flux_contribution_i_momentum_x.x);
                    flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.x + flux_contribution_i_momentum_y.x);
                    flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.x + flux_contribution_i_momentum_z.x);

                    factor = float(0.5f)*normal.y;
                    flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
                    flux_i_density_energy += factor*(ff_flux_contribution_density_energy.y+flux_contribution_i_density_energy.y);
                    flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.y + flux_contribution_i_momentum_x.y);
                    flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.y + flux_contribution_i_momentum_y.y);
                    flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.y + flux_contribution_i_momentum_z.y);

                    factor = float(0.5f)*normal.z;
                    flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
                    flux_i_density_energy += factor*(ff_flux_contribution_density_energy.z+flux_contribution_i_density_energy.z);
                    flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.z + flux_contribution_i_momentum_x.z);
                    flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.z + flux_contribution_i_momentum_y.z);
                    flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.z + flux_contribution_i_momentum_z.z);

                }
            }
            fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
            fluxes[i + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum.x;
            fluxes[i + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum.y;
            fluxes[i + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum.z;
            fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;

        }
    }
}

void time_step(int j, int nelr, float* old_variables, float* variables, float* step_factors, float* fluxes)
{
    #ifdef OMP_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for default(shared) schedule(auto)
    #elif defined(OMP_GPU_OFFLOAD_UM) && defined(MAP_ALL)
    #pragma omp target teams distribute parallel for
    #elif defined(OMP_GPU_OFFLOAD_UM)
    #pragma omp target teams distribute parallel for default(shared) schedule(auto) \
                is_device_ptr(old_variables) is_device_ptr(variables) \
                is_device_ptr(step_factors) is_device_ptr(fluxes)
    #else
    #pragma omp parallel for  default(shared) schedule(auto)
    #endif
    for(int blk = 0; blk < nelr/block_length; ++blk)
    {
        int b_start = blk*block_length;
        int b_end = (blk+1)*block_length > nelr ? nelr : (blk+1)*block_length;
        #if !defined(OMP_GPU_OFFLOAD) && !defined(OMP_GPU_OFFLOAD_UM)
        #pragma omp simd
        #endif
        for(unsigned long long i = b_start; i < b_end; ++i)
        {
            float factor = step_factors[i]/float(RK+1-j);

            variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
            variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
            variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];
            variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];
            variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];

        }
    }
}

/*
 * Main function
 */
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "specify data file name" << std::endl;
        return 0;
    }
    const char* data_file_name = argv[1];
    int num_iter = atoi(argv[2]);

#if defined (CUDA_UM)
    float *ff_variable;
    cudaMallocManaged((void**)&ff_variable, sizeof(float)*NVAR, cudaMemAttachGlobal);
#elif defined (OMP_GPU_OFFLOAD_UM)
    float *ff_variable = (float*) omp_target_alloc(sizeof(float)*NVAR, -100);
#else
    float *ff_variable = (float*) malloc(sizeof(float)*NVAR);
#endif
    float3d ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy;

    // set far field conditions
    {
        const float angle_of_attack = (3.1415926535897931 / 180.0f) * deg_angle_of_attack;

        ff_variable[VAR_DENSITY] = 1.4;

        float ff_pressure = float(1.0f);
        float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / ff_variable[VAR_DENSITY]);
        float ff_speed = float(ff_mach)*ff_speed_of_sound;

        float3d ff_velocity;
        ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
        ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
        ff_velocity.z = 0.0f;

        ff_variable[VAR_MOMENTUM+0] = ff_variable[VAR_DENSITY] * ff_velocity.x;
        ff_variable[VAR_MOMENTUM+1] = ff_variable[VAR_DENSITY] * ff_velocity.y;
        ff_variable[VAR_MOMENTUM+2] = ff_variable[VAR_DENSITY] * ff_velocity.z;

        ff_variable[VAR_DENSITY_ENERGY] = ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

        float3d ff_momentum;
        ff_momentum.x = *(ff_variable+VAR_MOMENTUM+0);
        ff_momentum.y = *(ff_variable+VAR_MOMENTUM+1);
        ff_momentum.z = *(ff_variable+VAR_MOMENTUM+2);
        compute_flux_contribution(ff_variable[VAR_DENSITY], ff_momentum, ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, ff_flux_contribution_momentum_x, 
                ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy);
    }
    int nel;
    int nelr;
    unsigned long long nelr_ll; // prevent overflow

    // read in domain geometry
    float* areas;
    int* elements_surrounding_elements;
    float* normals;
    {
        std::ifstream file;

        file.open(data_file_name);
		file >> nel;

        nelr = block_length*((nel*num_iter / block_length )+ std::min(1, (nel*num_iter) % block_length));
        nelr_ll = nelr; // prevent overflow

#if defined (CUDA_UM)
        cudaMallocManaged((void**)&areas, sizeof(float)*nelr_ll*NNB, cudaMemAttachGlobal);
        cudaMallocManaged((void**)&elements_surrounding_elements, sizeof(int)*nelr_ll*NNB, cudaMemAttachGlobal);
        cudaMallocManaged((void**)&normals, sizeof(float)*NDIM*NNB*nelr_ll, cudaMemAttachGlobal);
#elif defined (OMP_GPU_OFFLOAD_UM)
        areas = (float*) omp_target_alloc(sizeof(float)*nelr_ll*NNB, -100);
        elements_surrounding_elements = (int*) omp_target_alloc(sizeof(int)*nelr_ll*NNB, -100);
        normals = (float*) omp_target_alloc(sizeof(float)*NDIM*NNB*nelr_ll, -100);
#else
        areas = new float[nelr_ll];
        elements_surrounding_elements = new int[nelr_ll*NNB];
        normals = new float[NDIM*NNB*nelr_ll];
#endif

        for(unsigned long long i = 0; i < nel; i++)
        {
            file >> areas[i];
            for(unsigned long long j = 0; j < NNB; j++)
            {
                file >> elements_surrounding_elements[i + j*nelr];
                if(elements_surrounding_elements[i+j*nelr] < 0)
                    elements_surrounding_elements[i+j*nelr] = -1;
                elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering
    
                for(unsigned long long k = 0; k < NDIM; k++)
                {
                    file >>  normals[i + (j + k*NNB)*nelr];
                    normals[i + (j + k*NNB)*nelr] = -normals[i + (j + k*NNB)*nelr];
                }
            }
        }
        file.close();
        for(unsigned long long nn=1; nn<num_iter; nn++) {
    
            for(unsigned long long i = 0; i < nel; i++)
            {
                areas[i+nn*nel] = areas[i];
                for(unsigned long long j = 0; j < NNB; j++)
                {
                    elements_surrounding_elements[i + j*nelr + nn*nel] = elements_surrounding_elements[i + j*nelr];
                    for(unsigned long long k = 0; k < NDIM; k++)
                        normals[i + (j + k*NNB)*nelr + nn*nel] = normals[i + (j + k*NNB)*nelr];
                }
            }
        }
        nel *= num_iter;
        //printf("%d\n", nel);

        unsigned long long last = nel-1;

        for(unsigned long long i = nel; i < nelr; i++)
        {
            areas[i] = areas[last];
            for(unsigned long long j = 0; j < NNB; j++)
            {
                // duplicate the last element
                elements_surrounding_elements[i + j*nelr] = elements_surrounding_elements[last + j*nelr];
                for(unsigned long long k = 0; k < NDIM; k++) normals[i + (j + k*NNB)*nelr] = normals[last + (j + k*NNB)*nelr];
            }
        }
        // fill in remaining data
    }

    // Create arrays and set initial conditions
#if defined (CUDA_UM)
    float* variables;
    float* old_variables;
    float* fluxes;
    float* step_factors;
    cudaMallocManaged((void**)&variables, sizeof(float)*nelr_ll*NVAR, cudaMemAttachGlobal);
    cudaMallocManaged((void**)&old_variables, sizeof(float)*nelr_ll*NVAR, cudaMemAttachGlobal);
    cudaMallocManaged((void**)&fluxes, sizeof(float)*nelr_ll*NVAR, cudaMemAttachGlobal);
    cudaMallocManaged((void**)&step_factors, sizeof(float)*nelr_ll, cudaMemAttachGlobal);
#elif defined (OMP_GPU_OFFLOAD_UM)
    float* variables = (float*)omp_target_alloc(sizeof(float)*nelr_ll*NVAR, -100);
    float* old_variables = (float*)omp_target_alloc(sizeof(float)*nelr_ll*NVAR, -100);
    float* fluxes = (float*)omp_target_alloc(sizeof(float)*nelr_ll*NVAR, -100);
    float* step_factors = (float*)omp_target_alloc(sizeof(float)*nelr_ll, -100);
#else
    float* variables = alloc<float>(nelr_ll*NVAR);
    float* old_variables = alloc<float>(nelr_ll*NVAR);
    float* fluxes = alloc<float>(nelr_ll*NVAR);
    float* step_factors = alloc<float>(nelr_ll);
#endif
    initialize_variables(nelr, variables, ff_variable);

    // these need to be computed the first time in order to compute time step
    //std::cout << "Starting..." << std::endl;
#ifdef _OPENMP
    unsigned long total_size = sizeof(nelr) 
            + sizeof(float)*nelr*NNB        // area
            + sizeof(float)*nelr            // step_factors
            + sizeof(int)*nelr*NNB          // elements_surrounding_elements
            + sizeof(float)*NDIM*NNB*nelr   // normals
            + sizeof(float)*nelr*NVAR       // fluxes
            + sizeof(float)*NVAR            // ff_variable
            + sizeof(ff_flux_contribution_momentum_x) 
            + sizeof(ff_flux_contribution_momentum_y) 
            + sizeof(ff_flux_contribution_momentum_z) 
            + sizeof(ff_flux_contribution_density_energy) 
            + sizeof(float)*nelr*NVAR       // old_variables
            + sizeof(float)*nelr*NVAR;      // variables
    printf("Size: %f\n", (double)total_size / (1024.0 * 1024.0 * 1024.0));
    //std::cout << total_size << ",";
    double start = omp_get_wtime();
    #if defined(OMP_GPU_OFFLOAD)
    #pragma omp target data map(alloc: old_variables[0:(nelr*NVAR)]) \
                map(to: nelr, areas[0:nelr], step_factors[0:nelr], \
                    elements_surrounding_elements[0:(nelr*NNB)], \
                    normals[0:(NDIM*NNB*nelr)], \
                    fluxes[0:(nelr*NVAR)], \
                    ff_variable[0:NVAR], ff_flux_contribution_momentum_x, \
                    ff_flux_contribution_momentum_y, \
                    ff_flux_contribution_momentum_z, \
                    ff_flux_contribution_density_energy) \
                map(tofrom: variables[0:(nelr*NVAR)])
    #elif defined(OMP_GPU_OFFLOAD_UM) && defined(MAP_ALL)
    #pragma omp target data map(alloc: old_variables[0:(nelr*NVAR)]) \
                map(to: areas[0:nelr], step_factors[0:nelr], \
                    elements_surrounding_elements[0:(nelr*NNB)], \
                    normals[0:(NDIM*NNB*nelr)], \
                    fluxes[0:(nelr*NVAR)], \
                    ff_variable[0:NVAR], ff_flux_contribution_momentum_x, \
                    ff_flux_contribution_momentum_y, \
                    ff_flux_contribution_momentum_z, \
                    ff_flux_contribution_density_energy) \
                map(tofrom: variables[0:(nelr*NVAR)])
    #elif defined(OMP_GPU_OFFLOAD_UM)
    #pragma omp target data map(to: nelr, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z, \
                ff_flux_contribution_density_energy)
    #endif
#endif
    // Begin iterations
    for(int i = 0; i < iterations; i++)
    {
        copy<float>(old_variables, variables, nelr_ll*NVAR);

        // for the first iteration we compute the time step
        compute_step_factor(nelr, variables, areas, step_factors);

        for(int j = 0; j < RK; j++)
        {
            compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes, ff_variable, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy);
            time_step(j, nelr, old_variables, variables, step_factors, fluxes);
        }
    }

#ifdef _OPENMP
    double end = omp_get_wtime();
    std::cout  << "Compute time: " << (end-start) << std::endl;
    //std::cout << (end-start) << std::endl;
#endif


//    std::cout << "Saving solution..." << std::endl;
//    dump(variables, nel, nelr);
//    std::cout << "Saved solution..." << std::endl;


//    std::cout << "Cleaning up..." << std::endl;
#if defined (CUDA_UM)
    cudaFree(areas);
    cudaFree(elements_surrounding_elements);
    cudaFree(normals);

    cudaFree(variables);
    cudaFree(old_variables);
    cudaFree(fluxes);
    cudaFree(step_factors);
#elif defined (OMP_GPU_OFFLOAD_UM)
    omp_target_free(areas, omp_get_default_device());
    omp_target_free(elements_surrounding_elements, omp_get_default_device());
    omp_target_free(normals, omp_get_default_device());

    omp_target_free(variables, omp_get_default_device());
    omp_target_free(old_variables, omp_get_default_device());
    omp_target_free(fluxes, omp_get_default_device());
    omp_target_free(step_factors, omp_get_default_device());
#else 
    dealloc<float>(areas);
    dealloc<int>(elements_surrounding_elements);
    dealloc<float>(normals);

    dealloc<float>(variables);
    dealloc<float>(old_variables);
    dealloc<float>(fluxes);
    dealloc<float>(step_factors);
#endif

//    std::cout << "Done..." << std::endl;

    return 0;
}
