#pragma once
__global__ void Rho_InitKernel_IDIC (double* rho, double* rho1, double* rho2, int* Ind, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		rho[i] = 0.0;
		rho1[i] = 0.0;
		rho2[i] = 0.0;
	}
}

/*__global__ void Rho_dust_InitKernel_IDIC(double* dev_rho_dust[Num_of_dust_sort_max], int* dev_Ind_dust[Num_of_dust_sort_max], int Pm, int dust_id)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (dev_Ind_dust[dust_id][i] == 0)
	{
		dev_rho_dust[dust_id][i] = 0.0;
	}
}
*/

__global__ void Rho_dust_InitKernel_IDIC(double* rho, int* ind, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (ind[i] == 0)
	{
		rho[i] = 0.0;
	}
}

__global__ void Rho_Kernel_IDIC(double* x, double* y, double* z, double* rho, double* mas, int* Ind, int* Cell, int* Nn, int Pm, double Clx, double Cly, double Clz, double Clh, int Cnx, int Cny, int Cl, double h0, int k, int l, int g)
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


__global__ void Rho_dust_Kernel_IDIC(double** x, double** y, double** z, double** rho, double** mas, int** Ind, int** Cell, int** Nn, int Pm, double Clx, double Cly, double Clz, double Clh, int Cnx, int Cny, int Cl, double h0, int k, int l, int g, int dust_id)
{

	int j, Ni, Ci;
	double d, rho_tmp;

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[dust_id][i] == 0)
	{
		Ni = int((x[dust_id][i] - Clx) / Clh) + (int((y[dust_id][i] - Cly) / Clh)) * Cnx + (int((z[dust_id][i] - Clz) / Clh)) * Cnx * Cny;
		rho_tmp = 0.0;
		Ci = Ni + k + l * Cnx + g * Cnx * Cny;
		if ((Ci > -1) && (Ci < Cl))
		{
			j = Cell[dust_id][Ci];

			while (j > -1)
			{
				d = pow((x[dust_id][j] - x[dust_id][i]) * (x[dust_id][j] - x[dust_id][i]) + (y[dust_id][j] - y[dust_id][i]) * (y[dust_id][j] - y[dust_id][i]) + (z[dust_id][j] - z[dust_id][i]) * (z[dust_id][j] - z[dust_id][i]), 0.5);
				rho_tmp = rho_tmp + mas[dust_id][j] * W(d, h0);
				j = Nn[dust_id][j];
			}
		}

		rho[dust_id][i] = rho[dust_id][i] + rho_tmp;
	}
}

__global__ void t_stop_Kernel_IDIC(double* t_stop, double* p, double* rho, int* Ind, int Pm, int R_dust, double Gam_g)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		t_stop[i] = R_dust / (sqrt(Gam_g * p[i] / rho[i]) * rho[i]);

	}


}