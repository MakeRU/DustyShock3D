
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

#include "data.h"
#include "IDIC.h"

__device__ __host__ double W(double r, double h)
{
	const double Pi = 3.14159265;
	double k, c, tmp1;
	k = (double)fabs(r) / h;
	c = (double)1 / (Pi);
	if (k < 1.0) { tmp1 = (double)1.0 - 1.5 * k * k + 0.75 * k * k * k; }
	if ((k >= 1.0) && (k <= 2.0)) { tmp1 = (double)0.25 * (2 - k) * (2 - k) * (2 - k); }
	if (k > 2.0) { tmp1 = 0.0; }

	return  c / (h * h * h) * tmp1;
}

__device__ __host__ double dW(double r, double h)
{
	const double Pi = 3.14159265;
	double k, c, tmp1;
	k = (double) r / h;
	c = (double)1 / (Pi);
	if (k < -2.0) { tmp1 = 0.0; }
	if ((k >= -2.0) && (k <= -1.0)) { tmp1 = (double)0.75 * (2.0 + k) * (2.0 + k); }
	if ((k > -1.0) && (k < 0)) { tmp1 = (double)-3.0 * k - 2.25 * k * k; }
	if ((k >= 0) && (k <= 1.0)) { tmp1 = (double)-3.0 * k + 2.25 * k * k; }
	if ((k >= 1.0) && (k <= 2.0)) { tmp1 = (double)-0.75 * (2.0 - k) * (2.0 - k); }
	if (k > 2.0) { tmp1 = 0.0; }

	return  c / (h * h * h * h) * tmp1;
}

__device__ __host__ double Eq_t_stop(int l, double p, double rho, double gamma)
{
	return R_dust[l] / (sqrt(gamma * p / rho) * rho); 
}

__global__ void RhoInitKernel(double* rho, int* Ind, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		rho[i] = 0.0;
	}
}

__global__ void RhoKernel(double* x, double* y, double* z, double* rho, double* mas, int* Ind, int* Cell, int* Nn, int Pm, double Clx, double Cly, double Clz, double Clh, int Cnx, int Cny, int Cl, double h0, int k, int l, int g)
{

	int j, Ni, Ci;
	double d, rho_tmp;

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		Ni = int((x[i] - Clx) / Clh) + (int((y[i] - Cly) / Clh)) * Cnx + (int((z[i] - Clz) / Clh)) * Cnx * Cny;
		rho_tmp = 0.0;
		Ci = Ni + k + l * Cnx + g * Cnx * Cny;
		if ((Ci > -1) && (Ci < Cl))
		{
			j = Cell[Ci];

			while (j > -1)
			{
				d = pow((x[j] - x[i]) * (x[j] - x[i]) + (y[j] - y[i]) * (y[j] - y[i]) + (z[j] - z[i]) * (z[j] - z[i]), 0.5);
				rho_tmp = rho_tmp + mas[j] * W(d, h0);
				j = Nn[j];
			}
		}

		rho[i] = rho[i] + rho_tmp;
	}
}



__device__ __host__ double Eq_State(double rho, double e, int Type_of_state, double Gam_g, double Speed_of_sound_gas)
{
	//	return B*(pow(rho,Gam_liq)-1) + 1;
	if (Type_of_state == 0) { return rho * e * (Gam_g - 1); }
	if (Type_of_state == 1) { return Speed_of_sound_gas * Speed_of_sound_gas * rho; }
}

__device__ __host__ double Eq_State1(double p, double e, int Type_of_state, double Gam_g, double Speed_of_sound_gas)
{
	//	return pow((p-1)/B+1,1.0/Gam_liq);

	if (Type_of_state == 0) { return p / (e * (Gam_g - 1)); }
	if (Type_of_state == 1) { return p / (Speed_of_sound_gas * Speed_of_sound_gas); }
}

__global__ void PKernel(double* rho, double* p, double* e, int* Ind, int Pm, int Type_of_state, double Gam_g, double Speed_of_sound_gas)
{

	int k, l, g, j, Ni, Ci;
	double d, rho_tmp;

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		p[i] = Eq_State(rho[i], e[i], Type_of_state, Gam_g, Speed_of_sound_gas);
		
	}


}


__global__ void ForceInitKernel(double* Ax, double* Ay, double* Az, int* Ind, double* e_temp, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0 )
	{
		Ax[i] = 0.0;
		Ay[i] = 0.0;
		Az[i] = 0.0;
		e_temp[i] = 0.0;
	}
}


__global__ void ForceKernel(double* x, double* y, double* z, double* rho_gas, double* p_gas, double* mas_gas, double* e, double* Vx, double* Vy, double* Vz, double* Ax, double* Ay, double* Az, double* e_temp, int* Ind, int* Cell, int* Nn, int Pm, double Clx, double Cly, double Clz, double Clh, int Cnx, int Cny, int Cl, double h, double tau, double Gam_g, double alpha, double beta, double eps, int k, int l, int g)
{

	int j, Ni, Ci;
	double d, F_nu, nu1x, nu1y, nu1z, nu_temp, dist, A, F_tens, Cnu;

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{

		Ni = int((x[i] - Clx) / Clh) + (int((y[i] - Cly) / Clh)) * Cnx + (int((z[i] - Clz) / Clh)) * Cnx * Cny;
		Ci = Ni + k + l * Cnx + g * Cnx * Cny;
		if ((Ci > -1) && (Ci < Cl))
		{
			j = Cell[Ci];

			while (j > -1)
			{
				dist = pow((x[j] - x[i]) * (x[j] - x[i]) + (y[j] - y[i]) * (y[j] - y[i]) + (z[j] - z[i]) * (z[j] - z[i]), 0.5);
				if ((dist < 2 * h) && (dist > 0))
				{

					// Исск. вязкость

					F_nu = 0;
					nu1x = (Vx[i] - Vx[j]) * (x[i] - x[j]);
					nu1y = (Vy[i] - Vy[j]) * (y[i] - y[j]);
					nu1z = (Vz[i] - Vz[j]) * (z[i] - z[j]);
					nu_temp = nu1x + nu1y + nu1z;
					if (nu_temp < 0.0)
					{
						Cnu = 0.5 * (sqrt(Gam_g * p_gas[i] / rho_gas[i]) + sqrt(Gam_g * p_gas[j] / rho_gas[j]));
						nu_temp = h * nu_temp / (dist *dist + eps * h * h);
						F_nu = (-alpha * Cnu * nu_temp + beta * nu_temp * nu_temp) / (0.5 * (rho_gas[i] + rho_gas[j]));
					}

				//	F_tens = kapp * mas[j] * dist * W(dist, h0);
					//e_temp = e_temp + (mas_gas[i] * p_gas[i] / (rho_gas[i] * rho_gas[i]) + mas_gas[j] * p_gas[j] / (rho_gas[j] * rho_gas[j]) + mas_gas[j] / 2.0 * F_nu) * 
					//	((Vx[i] - Vx[j]) * dW(x[i] - x[j], h) + (Vy[i] - Vy[j]) * dW(y[i] - y[j], h) + (Vz[i] - Vz[j]) * dW(z[i] - z[j], h));
					// e_temp = e_temp + (mas_gas[i] * p_gas[i] / (rho_gas[i] * rho_gas[i]) + mas_gas[j] * p_gas[j] / (rho_gas[j] * rho_gas[j]) + mas_gas[j] / 2.0 * F_nu) *
					//	((Vx[i] - Vx[j]) * dW(x[i] - x[j], h) + (Vy[i] - Vy[j]) * dW(y[i] - y[j], h) + (Vz[i] - Vz[j]) * dW(z[i] - z[j], h));
					e_temp[i] = e_temp[i] + (mas_gas[i] * p_gas[i] / (rho_gas[i] * rho_gas[i]) + mas_gas[i] * F_nu / 2.0) *
						((Vx[i] - Vx[j]) * dW(x[i] - x[j], h) + (Vy[i] - Vy[j]) * dW(y[i] - y[j], h) + (Vz[i] - Vz[j]) * dW(z[i] - z[j], h));
					A = mas_gas[j] * (p_gas[i] / (rho_gas[i] * rho_gas[i]) + p_gas[j] / (rho_gas[j] * rho_gas[j]) + F_nu) * dW(dist, h);
					d = (x[j] - x[i]);
					Ax[i] = Ax[i] + d * A / dist;
					d = (y[j] - y[i]);
					Ay[i] = Ay[i] + d * A / dist;
					d = (z[j] - z[i]);
					Az[i] = Az[i] + d * A / dist;
				}



				j = Nn[j];
			}
		}

	}
}

__global__ void MoveKernel(double* x, double* y, double* z, double* Vx, double* Vy, double* Vz, double* Ax, double* Ay, double* Az, int* Ind, double* e_temp, double* e_gas, double tau, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;

	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		Vx[i] = Vx[i] + Ax[i] * tau;
		Vy[i] = Vy[i] + Ay[i] * tau;
		Vz[i] = Vz[i] + Az[i] * tau;
		e_gas[i] = e_gas[i] + e_temp[i] * tau;
 


		x[i] = x[i] + Vx[i] * tau;
		y[i] = y[i] + Vy[i]*tau;
		z[i] = z[i] + Vz[i]*tau;


	}

}



void Data_out(int num)
{
	FILE* out_file_gas, * out_file_dust, * out_file;
	char out_name[25];
	int i, j, l;
	double r,v, eps_out;


	eps_out = 1.0*h;

	if (num < 10000000) { sprintf(out_name, "Data/G%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/G0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/G00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/G000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/G0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/G00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/G000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
	//	if ((x_gas[i] >= 0.0) && (x_gas[i] <= 1.0))
		{
			r = sqrt(x_gas[i]* x_gas[i] + y_gas[i]*y_gas[i]+ z_gas[i]*z_gas[i]);
			v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
				x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_X%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_X0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_X00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_X000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_X0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_X00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_X000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if ((y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i] <= eps_out * eps_out))
		{
			r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
			v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
				x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_Y%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_Y0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_Y00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_Y000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_Y0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_Y00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_Y000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if ((x_gas[i] * x_gas[i] + z_gas[i] * z_gas[i] <= eps_out * eps_out))
		{
			r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
			v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
				x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_Z%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_Z0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_Z00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_Z000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_Z0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_Z00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_Z000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if ((x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] <= eps_out * eps_out))
		{
			r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
			v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
				x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_XY%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_XY0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_XY00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_XY000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_XY0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_XY00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_XY000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if (abs(z_gas[i]) <= eps_out)
		{
			r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
			v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
				x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_XZ%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_XZ0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_XZ00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_XZ000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_XZ0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_XZ00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_XZ000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if (abs(y_gas[i]) <= eps_out)
		{
		r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
		v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
		fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
			x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_YZ%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_YZ0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_YZ00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_YZ000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_YZ0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_YZ00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_YZ000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if (abs(x_gas[i]) <= eps_out)
		{
		r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
		v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
		fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
			x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);
}



void init_Ball()
{
	 double x_temp, y_temp, z_temp;
	 
	p = -1;
	Im = int((X_max - X_min) * Particle_on_length);
	Jm = int((Y_max - Y_min) * Particle_on_length);
	Km = int((Z_max - Z_min) * Particle_on_length);

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = X_min + (double)(i * dlh);
				y_temp = Y_min + (double)(j * dlh);
				z_temp = Z_min + (double)(k * dlh);
				if (x_temp * x_temp + y_temp * y_temp + z_temp * z_temp <= 1.0) {
				p = p + 1;
				x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
				y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
				z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
				mas_gas[p] = 1.0 / (Particle_on_length * Particle_on_length * Particle_on_length);
				Vx_gas[p] = 0.0;
				Vy_gas[p] = 0.0;
				Vz_gas[p] = 0.0;
				Ax_gas[p] = 0.0;
				Ay_gas[p] = 0.0;
				Az_gas[p] = 0.0;
				e_gas[p] = 2.5;
				p_gas[p] = 1.0;
				rho_gas[p] = 1.0;
				Ind_gas[p] = 0;
				};
			}

	Pr = p;

	Pm = p;
}

void init_Sod_X()
{
	double x_temp, y_temp, z_temp;
	double p_left, p_right, e_left, e_right, rho_left, rho_right;
	
	p = -1;
	Im = int((X_max - X_min) * Particle_on_length);
	Jm = int((Y_max - Y_min) * Particle_on_length);
	Km = int((Z_max - Z_min) * Particle_on_length);

	p_left = 1.0;
	p_right = 0.1;
	e_left = 2.5;
	e_right = 2.0;
	rho_left = Eq_State1(p_left, e_left, Type_of_state, 1.4, 1.0);
	rho_right = Eq_State1(p_right, e_right, Type_of_state, 1.4, 1.0);
	

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = X_min + (double)(i * dlh);
				y_temp = Y_min + (double)(j * dlh);
				z_temp = Z_min + (double)(k * dlh);
				if ((abs(x_temp) <= 1.0) && (abs(y_temp) <= 1.0) && (abs(z_temp) <= 1.0))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					if (x_temp <= 0.0)
					{
						e_gas[p] = e_left;
						p_gas[p] = p_left;
						rho_gas[p] = rho_left;
						mas_gas[p] = rho_left / (Particle_on_length * Particle_on_length * Particle_on_length);
					}
					else
					{
						e_gas[p] = e_right;
						p_gas[p] = p_right;
						rho_gas[p] = rho_right;
						mas_gas[p] = rho_right / (Particle_on_length * Particle_on_length * Particle_on_length);
					}
					if ((abs(x_temp) <= 0.8) && (abs(y_temp) <= 0.8) && (abs(z_temp) <= 0.8))
					{
						Ind_gas[p] = 0;
					}
					else
					{
						Ind_gas[p] = 1;
					}
					

				};
			}

	Pr = p;

	Pm = p;
}

void init_Sod_X_mas()
{
	double x_temp, y_temp, z_temp;
	double p_left, p_right, e_left, e_right, rho_left, rho_right, dlh_left, dlh_right, mas_particle;
	
	double border_length = 2.5 * h;


	p_left = 1.0;
	p_right = 0.1;
	e_left = 2.5;
	e_right = 2.0;
	rho_left = Eq_State1(p_left, e_left, Type_of_state, 1.4, 1.0);
	rho_right = Eq_State1(p_right, e_right, Type_of_state, 1.4, 1.0);

	mas_particle = 1.0 / (Particle_on_length * Particle_on_length * Particle_on_length);
	dlh_left = pow(mas_particle/rho_left,1/3.0);
	dlh_right = pow(mas_particle / rho_right, 1 / 3.0);

	p = -1;

	Im = int((0.0 - X_min) / dlh_left) +1;
	Jm = int((Y_max - Y_min) / dlh_left) +1;
	Km = int((Z_max - Z_min) / dlh_left) +1;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = 0.0 - (double)(i * dlh_left);
				y_temp = Y_min + (double)(j * dlh_left);
				z_temp = Z_min + (double)(k * dlh_left);
				if ((abs(x_temp) <= 1.0) && (abs(y_temp) <= 1.0) && (abs(z_temp) <=1.0))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = e_left;
					p_gas[p] = p_left;
					rho_gas[p] = rho_left;
					mas_gas[p] = mas_particle;
					}
				if ((abs(x_temp) <= 1.0 - border_length) && (abs(y_temp) <= 1.0 - border_length) && (abs(z_temp) <= 1.0 - border_length))
					{
						Ind_gas[p] = 0;
					}
				else
					{
						Ind_gas[p] = 1;
					}


				};

	Im = int((X_max - 0.0) / dlh_right) +1;
	Jm = int((Y_max - Y_min) / dlh_right) +1;
	Km = int((Z_max - Z_min) / dlh_right) +1;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = dlh_right + (double)(i * dlh_right);
				y_temp = Y_min + (double)(j * dlh_right);
				z_temp = Z_min + (double)(k * dlh_right);
				if ((abs(x_temp) <= 1.0) && (abs(y_temp) <= 1.0) && (abs(z_temp) <= 1.0))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = e_right;
					p_gas[p] = p_right;
					rho_gas[p] = rho_right;
					mas_gas[p] = mas_particle;
				}
				if ((abs(x_temp) <= 1.0 - border_length) && (abs(y_temp) <= 1.0 - border_length) && (abs(z_temp) <= 1.0 - border_length))
				{
					Ind_gas[p] = 0;
				}
				else
				{
					Ind_gas[p] = 1;
				}

			};

	
	Pr = p;

	Pm = p;
}

void init_Sod_X_e()
{
	double x_temp, y_temp, z_temp;
	double p_left, p_right, e_left, e_right, rho_left, rho_right, dlh_left, dlh_right, mas_particle;

	double border_length = 2.5 * h;
	double X_cube = 1.0, Y_cube = 0.8, Z_cube = 0.8;

	e_left = 2.5;
	e_right = 2.0;
	rho_left = 1.0;  
	rho_right = 1.0;  
	p_left = Eq_State(rho_left, e_left, Type_of_state, 1.4, 1.0);
	p_right = Eq_State(rho_right, e_right, Type_of_state, 1.4, 1.0);


	mas_particle = 1.0 / (Particle_on_length * Particle_on_length * Particle_on_length);
	dlh_left = pow(mas_particle / rho_left, 1 / 3.0);
	dlh_right = pow(mas_particle / rho_right, 1 / 3.0);

	p = -1;

	Im = int((0.0 - X_min) / dlh_left) + 1;
	Jm = int((Y_max - Y_min) / dlh_left) + 1;
	Km = int((Z_max - Z_min) / dlh_left) + 1;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = 0.0 - (double)(i * dlh_left);
				y_temp = Y_min + (double)(j * dlh_left);
				z_temp = Z_min + (double)(k * dlh_left);
				if ((abs(x_temp) <= X_cube) && (abs(y_temp) <= Y_cube) && (abs(z_temp) <= Z_cube))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = e_left;
					p_gas[p] = p_left;
					rho_gas[p] = rho_left;
					mas_gas[p] = mas_particle;
				}
				if ((abs(x_temp) <= X_cube - border_length) && (abs(y_temp) <= X_cube - border_length) && (abs(z_temp) <= Z_cube - border_length))
				{
					Ind_gas[p] = 0;
				}
				else
				{
					Ind_gas[p] = 1;
				}


			};

	Im = int((X_max - 0.0) / dlh_right) + 1;
	Jm = int((Y_max - Y_min) / dlh_right) + 1;
	Km = int((Z_max - Z_min) / dlh_right) + 1;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = dlh_right + (double)(i * dlh_right);
				y_temp = Y_min + (double)(j * dlh_right);
				z_temp = Z_min + (double)(k * dlh_right);
				if ((abs(x_temp) <= X_cube) && (abs(y_temp) <= Y_cube) && (abs(z_temp) <= Z_cube))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = e_right;
					p_gas[p] = p_right;
					rho_gas[p] = rho_right;
					mas_gas[p] = mas_particle;
				}
				if ((abs(x_temp) <= X_cube - border_length) && (abs(y_temp) <= X_cube - border_length) && (abs(z_temp) <= Z_cube - border_length))
				{
					Ind_gas[p] = 0;
				}
				else
				{
					Ind_gas[p] = 1;
				}

			};


	Pr = p;

	Pm = p;
}


void init_Dust_Ball()
{
	double x_temp, y_temp, z_temp;
	int dust_id;

	p = -1;
	Im = int((X_max - X_min) * Particle_on_length);
	Jm = int((Y_max - Y_min) * Particle_on_length);
	Km = int((Z_max - Z_min) * Particle_on_length);

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = X_min + (double)(i * dlh);
				y_temp = Y_min + (double)(j * dlh);
				z_temp = Z_min + (double)(k * dlh);
				if (x_temp * x_temp + y_temp * y_temp + z_temp * z_temp <= 1.0) {
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					mas_gas[p] = 1.0 / (Particle_on_length * Particle_on_length * Particle_on_length);
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = 2.5;
					p_gas[p] = 1.0;
					rho_gas[p] = 1.0;
					Ind_gas[p] = 0;
				};
			}

	Pr = p;

	Pm = p;

	Num_dust_sort = 2;
	mas_gas_dust[0] = 0.5;
	R_dust[0] = 1.0;
	mas_gas_dust[1] = 0.3333;
	R_dust[1] = 0.001;


	for (dust_id = 0; dust_id < Num_dust_sort; dust_id++)
	{
		x_dust[dust_id] = new double[Pm + 1];
		y_dust[dust_id] = new double[Pm + 1];
		z_dust[dust_id] = new double[Pm + 1];
		h_dust[dust_id] = new double[Pm + 1];
		mas_dust[dust_id] = new double[Pm + 1];
		rho_dust[dust_id] = new double[Pm + 1];
		Vx_dust[dust_id] = new double[Pm + 1];
		Vy_dust[dust_id] = new double[Pm + 1];
		Vz_dust[dust_id] = new double[Pm + 1];
		Ax_dust[dust_id] = new double[Pm + 1];
		Ay_dust[dust_id] = new double[Pm + 1];
		Az_dust[dust_id] = new double[Pm + 1];
		ind_dust[dust_id] = new int[Pm + 1];
		t_stop[dust_id] = new double[Pm + 1];
		N_dust[dust_id] = new int[Pm + 1]; // Number of next particle in cell
	}

	
	for (dust_id = 0; dust_id < Num_dust_sort; dust_id++)
	{
		for (p = 0; p <= Pm; p++)
		{
			
			x_dust[dust_id][p] = x_gas[p];
			y_dust[dust_id][p] = y_gas[p]; 
			z_dust[dust_id][p] = z_gas[p];
			h_dust[dust_id][p] = h;
			mas_dust[dust_id][p] = mas_gas[p]* mas_gas_dust[dust_id]; 
			rho_dust[dust_id][p] = rho_gas[p] * mas_gas_dust[dust_id]; 
			Vx_dust[dust_id][p] = 0.0;
			Vy_dust[dust_id][p] = 0.0;
			Vz_dust[dust_id][p] = 0.0;
			Ax_dust[dust_id][p] = 0.0;
			Ay_dust[dust_id][p] = 0.0;
			Az_dust[dust_id][p] = 0.0;
			ind_dust[dust_id][p] = Ind_gas[p];
			t_stop[dust_id][p] = Eq_t_stop(dust_id, p_gas[p], rho_gas[p], Gam_g);

		}
	}

	Clh = Coeff_h_cell * h; // Cell length 
	Cnx = int((Cmx - Clx) / Clh);
	Cny = int((Cmy - Cly) / Clh);
	Cnz = int((Cmz - Clz) / Clh);
	
	Number_of_dust_cell = (Cnx + 1) * (Cny + 1) * (Cnz + 1); 
	x_dust_cell = new double[Number_of_dust_cell + 1];
	y_dust_cell = new double[Number_of_dust_cell + 1];
	z_dust_cell = new double[Number_of_dust_cell + 1];
	v_g_average = new double[Number_of_dust_cell + 1];
	rho_g_average = new double[Number_of_dust_cell + 1];
	e_g_average = new double[Number_of_dust_cell + 1];
	g_average_count = new int[Number_of_dust_cell + 1];
	Psi_av_new = new double[Number_of_dust_cell + 1];
	v_av_new = new double[Number_of_dust_cell + 1];
	y_av_new = new double[Number_of_dust_cell + 1];
	y_av = new double[Number_of_dust_cell + 1];
	beta_cell = new double[Number_of_dust_cell + 1];

	for (dust_id = 0; dust_id < Num_dust_sort; dust_id++)
	{
		v_d_average[dust_id] = new double[Number_of_dust_cell + 1];
		rho_d_average[dust_id] = new double[Number_of_dust_cell + 1];
		eps_cell[dust_id] = new double[Number_of_dust_cell + 1];
		d_average_count[dust_id] = new int[Number_of_dust_cell + 1];
		t_stop_average[dust_id] = new double[Number_of_dust_cell + 1];
		x_av_new[dust_id] = new double[Number_of_dust_cell + 1];
		x_av[dust_id] = new double[Number_of_dust_cell + 1];
		u_av_new[dust_id] = new double[Number_of_dust_cell + 1];
		b_cell[dust_id] = new double[Number_of_dust_cell + 1];
	}

	cell_dust_width = Coeff_h_cell * h; 
	for (i = 0; i <= Number_of_dust_cell; i++)
	{
		//x_dust_cell[i] = (double) X_min + i * cell_dust_width;
		//x_dust_cell[i] = (double) Y_min + i * cell_dust_width;
		//z_dust_cell[i] = (double) Y_min + i * cell_dust_width;
		v_g_average[i] = 0.0;
		rho_g_average[i] = 0.0;
		e_g_average[i] = 0.0;
		g_average_count[i] = 0;
		Psi_av_new[i] = 0.0;
		v_av_new[i] = 0.0;
		y_av_new[i] = 0.0;
		y_av[i] = 0.0;
		for (dust_id = 0; dust_id < Num_dust_sort; dust_id++)
		{
			v_d_average[dust_id][i] = 0.0;
			rho_d_average[dust_id][i] = 0.0;
			eps_cell[dust_id][i] = 0.0;
			t_stop_average[dust_id][i] = 0.0;
			d_average_count[dust_id][i] = 0;
			x_av_new[dust_id][i] = 0.0;
			x_av[dust_id][i] = 0.0;
			u_av_new[dust_id][i] = 0.0;
			b_cell[dust_id][i] = 0.0;
		}
	}

}


int main()

{


	FILE *ini_file;
	char s[128];
//	fopen_s(&ini_file, "Init.txt", "r");
	ini_file = fopen("Init.txt", "r");
	fgets(s, 128, ini_file);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &X_min);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &X_max);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Y_min);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Y_max);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Z_min);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Z_max);
	fgets(s, 128, ini_file); sscanf(s, "%d", &Type_of_state);
	fgets(s, 128, ini_file); sscanf(s, "%d", &Maximum_particle);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Particle_on_length);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &h);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &h_cell);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &tau);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &T_end);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &T_out);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &alpha);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &beta);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &eps);


/*	fgets(s, 128, ini_file);	sscanf(s, "%lf", &Zm);
	fgets(s, 128, ini_file);	sscanf(s, "%lf", &c0);
	fgets(s, 128, ini_file);	sscanf(s, "%lf", &n);
	
	fgets(s, 128, ini_file); sscanf(s, "%lf", &p_in);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &p_out);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &tau);
	fgets(s, 128, ini_file);	sscanf(s, "%d", &Te);
	fgets(s, 128, ini_file);	sscanf(s, "%d", &File_int);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Geps);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &alpha);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &beta);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &eps);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &sigm);	
*/

	fclose(ini_file);

	Speed_of_sound_gas = 1.0;
	dlh = (double) 1.0 / Particle_on_length;
	Tm = 0.0;
	out_num = 0;

	// Memory

	x_gas = new double[Maximum_particle];
	y_gas = new double[Maximum_particle];
	z_gas = new double[Maximum_particle];
	p_gas = new double[Maximum_particle];
	rho_gas = new double[Maximum_particle];
	mas_gas = new double[Maximum_particle];
	Vx_gas = new double[Maximum_particle];
	Vy_gas = new double[Maximum_particle];
	Vz_gas = new double[Maximum_particle];
	Ax_gas = new double[Maximum_particle];
	Ay_gas = new double[Maximum_particle];
	Az_gas = new double[Maximum_particle];
	e_gas = new double[Maximum_particle];
	Ind_gas = new int[Maximum_particle];
	Nn = new int[Maximum_particle]; // Number of next particle in cell

	Clx = X_min;
	Cly = Y_min;
	Clz = Z_min;
	Cmx = X_max;
	Cmy = Y_max;
	Cmz = Z_max;
	Clh = 2.0 * h; // Cell length 
	Cnx = int((Cmx - Clx) / Clh);
	Cny = int((Cmy - Cly) / Clh);
	Cnz = int((Cmz - Clz) / Clh);

	Cl = (Cnx + 1) * (Cny + 1) * (Cnz + 1);
	Cell_gas = new int[Cl + 1]; // Number of first particle
	Ne = new int[Cl + 1]; //  Number of last particle



	// Particle placing

	// init_Ball(); 
	//  init_Sod_X();
	// init_Sod_X_mas();
	// init_Sod_X_e();
	init_Dust_Ball();


	cudaSetDevice(1);

	cudaMalloc((void**)&dev_x_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_y_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_z_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Vx_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Vy_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Vz_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Ax_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Ay_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Az_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_rho_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_p_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_e_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_mas_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Ind_gas, (Pm + 1) * sizeof(int));
	cudaMalloc((void**)&dev_Cell_gas, (Cl) * sizeof(int));
	cudaMalloc((void**)&dev_Nn_gas, (Pm + 1) * sizeof(int));
	cudaMalloc((void**)&dev_e_temp, (Pm + 1) * sizeof(double));
	
	for (int dust_id = 0; dust_id < Num_dust_sort; dust_id++)
	{
		cudaMalloc((void**)&dev_x_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_y_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_z_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vx_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vy_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vz_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Ax_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Ay_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Az_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_rho_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_mas_dust[dust_id], (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Ind_dust[dust_id], (Pm + 1) * sizeof(int));
		cudaMalloc((void**)&dev_Cell_dust[dust_id], (Cl) * sizeof(int));
		cudaMalloc((void**)&dev_Nn_dust[dust_id], (Pm + 1) * sizeof(int));

	}


	// cudaMemcpyToSymbol(&Gam_g, &Gam_g, sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_Gam_g, sizeof(double));
	cudaMemcpy(dev_Gam_g, &Gam_g, sizeof(double), cudaMemcpyHostToDevice);




	dim3 gridSize = dim3(Pm / 256 + 1, 1, 1);  
	dim3 blockSize = dim3(16, 16, 1);

	Tm = 0.0;
	out_num = 0;

	Data_out(out_num);
	out_num = out_num + 1;


	do
	{
		Tm = Tm + tau;
		printf("Time %5.3lf \n", Tm);


		// Allocation particled to cells

		for (i = 0; i <= Cl; i++)
		{
			Cell_gas[i] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			Nn_gas[p] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			i = int((x_gas[p] - Clx) / Clh) + (int((y_gas[p] - Cly) / Clh)) * Cnx + (int((z_gas[p] - Clz) / Clh)) * Cnx * Cny;
			if ((i >= 0) && (i <= Cl))
			{
				if (Cell_gas[i] == -1) { Cell_gas[i] = p; Ne_gas[i] = p; Nn_gas[p] = -1; }
				else { Nn_gas[Ne_gas[i]] = p; Ne_gas[i] = p; Nn_gas[p] = -1; };
			}
		}

		for (int dust_id = 0; dust_id < Num_dust_sort; dust_id++)
		{

			for (i = 0; i <= Cl; i++)
			{
				Cell_dust[dust_id][i] = -1;
			}

			for (p = 0; p <= Pm; p++)
			{
				Nn_dust[dust_id][p] = -1;
			}

			for (p = 0; p <= Pm; p++)
			{
				i = int((x_dust[dust_id][p] - Clx) / Clh) + (int((y_dust[dust_id][p] - Cly) / Clh)) * Cnx + (int((z_dust[dust_id][p] - Clz) / Clh)) * Cnx * Cny;
				if ((i >= 0) && (i <= Cl))
				{
					if (Cell_dust[dust_id][i] == -1) { Cell_dust[dust_id][i] = p; Ne_dust[dust_id][i] = p; Nn_dust[dust_id][p] = -1; }
					else { Nn_dust[dust_id][Ne_dust[dust_id][i]] = p; Ne_dust[dust_id][i] = p; Nn_dust[dust_id][p] = -1; };
				}
			}
		}


		cudaMemcpy(dev_x_gas, x_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_y_gas, y_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_z_gas, z_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_rho_gas, rho_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_p_gas, p_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mas_gas, mas_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_e_gas, e_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vx_gas, Vx_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vy_gas, Vy_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vz_gas, Vz_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Ax_gas, Ax_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Ay_gas, Ay_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Az_gas, Az_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Ind_gas, Ind_gas, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Cell_gas, Cell_gas, (Cl) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Nn_gas, Nn_gas, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);

		Rho_gas_InitKernel_IDIC << <gridSize, blockSize >> > (dev_rho_gas, dev_Ind_gas, Pm);
		cudaDeviceSynchronize();

		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{

					Rho_gas_Kernel_IDIC << <gridSize, blockSize >> > (dev_x_gas, dev_y_gas, dev_z_gas, dev_rho_gas, dev_mas_gas, dev_Ind_gas, dev_Cell, dev_Nn, Pm, Clx, Cly, Clz, Clh, Cnx, Cny, Cl, h, k, l, g);
					cudaDeviceSynchronize();
				}

		PKernel << <gridSize, blockSize >> > (dev_rho_gas, dev_p_gas, dev_e_gas, dev_Ind_gas, Pm, Type_of_state, Gam_g, Speed_of_sound_gas);
		cudaDeviceSynchronize();


		for (int dust_id = 0; dust_id < Num_dust_sort; dust_id++)
		{
			Rho_dust_InitKernel_IDIC << <gridSize, blockSize >> > (dev_rho_dust[dust_id], dev_Ind_dust[dust_id], Pm);
			cudaDeviceSynchronize();
		}


		for (int dust_id = 0; dust_id < Num_dust_sort; dust_id++)
		{
			for (k = -1; k <= 1; k++)
				for (l = -1; l <= 1; l++)
					for (g = -1; g <= 1; g++)
					{

						Rho_dust_Kernel_IDIC << <gridSize, blockSize >> > (dev_x_dust[dust_id], dev_y_dust[dust_id], dev_z_dust[dust_id], dev_rho_dust[dust_id], dev_mas_dust[dust_id], dev_Ind_dust[dust_id], dev_Cell_dust[dust_id], dev_Nn_dust[dust_id], Pm, Clx, Cly, Clz, Clh, Cnx, Cny, Cl, h, k, l, g);
						cudaDeviceSynchronize();
					}
		}

		for (int dust_id = 0; dust_id < Num_dust_sort; dust_id++)
		{
			t_stop_Kernel_IDIC << <gridSize, blockSize >> > (dev_t_stop[dust_id], dev_p_gas, dev_rho_gas, dev_Ind_dust[dust_id], Pm, R_dust[dust_id], Gam_g);
			cudaDeviceSynchronize();
		}



		ForceInitKernel << <gridSize, blockSize >> > (dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_Ind_gas, dev_e_temp, Pm);
		cudaDeviceSynchronize();

		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{
					ForceKernel << <gridSize, blockSize >> > (dev_x_gas, dev_y_gas, dev_z_gas, dev_rho_gas, dev_p_gas, dev_mas_gas, dev_e_gas, dev_Vx_gas, dev_Vy_gas, dev_Vz_gas, dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_e_temp, dev_Ind_gas, dev_Cell, dev_Nn, Pm, Clx, Cly, Clz, Clh, Cnx, Cny, Cl, h, tau, Gam_g, alpha, beta, eps, k, l, g);
					cudaDeviceSynchronize();
				}


		MoveKernel << <gridSize, blockSize >> > (dev_x_gas, dev_y_gas, dev_z_gas, dev_Vx_gas, dev_Vy_gas, dev_Vz_gas, dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_Ind_gas, dev_e_temp, dev_e_gas, tau, Pm);
		cudaDeviceSynchronize();



		cudaMemcpy(x_gas, dev_x_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(y_gas, dev_y_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(z_gas, dev_z_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(rho_gas, dev_rho_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(p_gas, dev_p_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(mas_gas, dev_mas_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(e_gas, dev_e_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vx_gas, dev_Vx_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vy_gas, dev_Vy_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vz_gas, dev_Vz_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ax_gas, dev_Ax_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ay_gas, dev_Ay_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Az_gas, dev_Az_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);


		if (Tm >= out_num * T_out)
		{
			Data_out(out_num);
			out_num = out_num + 1;
		}

	} while (Tm < T_end);

	/*
	do
	{
		Tm = Tm + tau;
		printf("Time %5.3lf \n", Tm);


		// Allocation particled to cells

		for (i = 0; i <= Cl; i++)
		{
			Cell[i] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			Nn[p] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			i = int((x_gas[p] - Clx) / Clh) + (int((y_gas[p] - Cly) / Clh)) * Cnx + (int((z_gas[p] - Clz) / Clh)) * Cnx * Cny;
			if ((i >= 0) && (i <= Cl))
			{
				if (Cell[i] == -1) { Cell[i] = p; Ne[i] = p; Nn[p] = -1; }
				else { Nn[Ne[i]] = p; Ne[i] = p; Nn[p] = -1; };
			}
		}


		cudaMemcpy(dev_x_gas, x_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_y_gas, y_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_z_gas, z_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_rho_gas, rho_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_p_gas, p_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mas_gas, mas_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_e_gas, e_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vx_gas, Vx_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vy_gas, Vy_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vz_gas, Vz_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Ax_gas, Ax_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Ay_gas, Ay_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Az_gas, Az_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Ind_gas, Ind_gas, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Cell, Cell, (Cl) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Nn, Nn, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);
		
		RhoInitKernel <<<gridSize, blockSize >>> (dev_rho_gas, dev_Ind_gas, Pm);
		cudaDeviceSynchronize();

		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{

					RhoKernel <<<gridSize, blockSize >>> (dev_x_gas, dev_y_gas, dev_z_gas, dev_rho_gas, dev_mas_gas, dev_Ind_gas, dev_Cell, dev_Nn, Pm, Clx, Cly, Clz, Clh, Cnx, Cny, Cl, h, k, l, g);
					cudaDeviceSynchronize();
				}

		PKernel <<<gridSize, blockSize >>> (dev_rho_gas, dev_p_gas, dev_e_gas, dev_Ind_gas, Pm, Type_of_state, Gam_g, Speed_of_sound_gas);
		cudaDeviceSynchronize();


		ForceInitKernel <<<gridSize, blockSize >>> (dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_Ind_gas, dev_e_temp, Pm);
		cudaDeviceSynchronize();

		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{
					ForceKernel <<<gridSize, blockSize >>> (dev_x_gas, dev_y_gas, dev_z_gas, dev_rho_gas, dev_p_gas, dev_mas_gas, dev_e_gas, dev_Vx_gas, dev_Vy_gas, dev_Vz_gas, dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_e_temp, dev_Ind_gas, dev_Cell, dev_Nn, Pm, Clx, Cly, Clz, Clh, Cnx, Cny, Cl, h, tau, Gam_g, alpha, beta, eps, k, l, g);
					cudaDeviceSynchronize();
				}


		MoveKernel <<<gridSize, blockSize >>> (dev_x_gas, dev_y_gas, dev_z_gas, dev_Vx_gas, dev_Vy_gas, dev_Vz_gas, dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_Ind_gas, dev_e_temp, dev_e_gas, tau, Pm);
		cudaDeviceSynchronize();



		cudaMemcpy(x_gas, dev_x_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(y_gas, dev_y_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(z_gas, dev_z_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(rho_gas, dev_rho_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(p_gas, dev_p_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(mas_gas, dev_mas_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(e_gas, dev_e_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vx_gas, dev_Vx_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vy_gas, dev_Vy_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vz_gas, dev_Vz_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ax_gas, dev_Ax_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ay_gas, dev_Ay_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Az_gas, dev_Az_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		
		
		if (Tm >= out_num * T_out)
		{
			Data_out(out_num);
			out_num = out_num + 1;
		}

	} while (Tm < T_end);
	
	*/

    return 0;
}

