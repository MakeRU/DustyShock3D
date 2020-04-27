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

__global__ void t_stop_Kernel_IDIC(double* t_stop, double* p, double* rho, int* Ind, int Pm, double R_dust, double Gam_g)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		t_stop[i] = R_dust / (sqrt(Gam_g * p[i] / rho[i]) * rho[i]);

	}


}

__global__ void EnergyInitKernel_IDIC (double *e_temp, int *Ind, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		e_temp[i] = 0.0;
	}
}

__global__ void PsiInitKernel_IDIC(double* Psi_av_x, double* Psi_av_y, double* Psi_av_z, int Number_of_average_cell)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i >  Number_of_average_cell) return;

	Psi_av_x[i] = 0.0;
	Psi_av_y[i] = 0.0;
	Psi_av_z[i] = 0.0;
}

__global__ void PsiKernel_IDIC(double* x, double* y, double* z, double* Vx, double* Vy, double* Vz, double* rho_gas, double* p_gas, double* mas_gas, int* Ind, int* Cell, int* Nn, double* e_temp, double* Psi_av_new_x,
	double* Psi_av_new_y, double* Psi_av_new_z, int Pm, double Clx, double Cly, double Clz, double Clh, int Cnx, int Cny, int Cl, double h, double X_min, double Y_min, double Z_min, double  average_cell_width, 
	int  Number_of_average_cell_x, int  Number_of_average_cell_y, int Number_of_average_cell, double Gam_g, double alpha, double beta, double eps, int k, int l, int g)
{

	int j, Ni, Ci, cell_num;
	double d, F_nu, nu1x, nu1y, nu1z, nu_temp, dist, Grad, Cnu, A;

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
						nu_temp = h * nu_temp / (dist * dist + eps * h * h);
						F_nu = (-alpha * Cnu * nu_temp + beta * nu_temp * nu_temp) / (0.5 * (rho_gas[i] + rho_gas[j]));
					}

					cell_num = int((x[i] - X_min) / average_cell_width) + (int((y[i] - Y_min) / average_cell_width)) * Number_of_average_cell_x + (int((z[i] - Z_min) / average_cell_width)) * Number_of_average_cell_x * Number_of_average_cell_y;
					Grad = dW(dist, h) / dist;
					if ((cell_num > -1) && (cell_num < Number_of_average_cell)) 
					{
						A = mas_gas[j] * (p_gas[i] / (rho_gas[i] * rho_gas[i]) + p_gas[j] / (rho_gas[j] * rho_gas[j]) + F_nu) * dW(dist, h);
						d = (x[j] - x[i]);
						Psi_av_new_x[cell_num] = Psi_av_new_x[cell_num] + d * A / dist;
						d = (y[j] - y[i]);
						Psi_av_new_y[cell_num] = Psi_av_new_y[cell_num] + d * A / dist;
						d = (z[j] - z[i]);
						Psi_av_new_z[cell_num] = Psi_av_new_z[cell_num] + d * A / dist;

					//	Psi_av_new_x[cell_num] = Psi_av_new_x[cell_num] + mas_gas[j] * (p_gas[i] / (rho_gas[i] * rho_gas[i]) + p_gas[j] / (rho_gas[j] * rho_gas[j]) + F_nu) * (x[j] - x[i]) * Grad;
					//	Psi_av_new_y[cell_num] = Psi_av_new_y[cell_num] + mas_gas[j] * (p_gas[i] / (rho_gas[i] * rho_gas[i]) + p_gas[j] / (rho_gas[j] * rho_gas[j]) + F_nu) * (y[j] - y[i]) * Grad;
					//	Psi_av_new_z[cell_num] = Psi_av_new_z[cell_num] + mas_gas[j] * (p_gas[i] / (rho_gas[i] * rho_gas[i]) + p_gas[j] / (rho_gas[j] * rho_gas[j]) + F_nu) * (z[j] - z[i]) * Grad;
					}
					e_temp[i] = e_temp[i] + (mas_gas[i] * p_gas[i] / (rho_gas[i] * rho_gas[i])  + mas_gas[i] / 2.0 * F_nu)* ( nu1x + nu1y + nu1z) * Grad;
				}

			j = Nn[j];
			}
		}

	}
}


__global__ void Force_InitKernel_IDIC(double* Ax, double* Ay, double* Az, int* Ind, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		Ax[i] = 0.0;
		Ay[i] = 0.0;
		Az[i] = 0.0;
	}
}


__global__ void ForceKernel_IDIC(double* x, double* y, double* z, double* rho_gas, double* p_gas, double* mas_gas, double* e_temp, double* Vx, double* Vy, double* Vz, double* Ax, double* Ay, double* Az, int* Ind, int* Cell, int* Nn, int Pm, double Clx, double Cly, double Clz, double Clh, int Cnx, int Cny, int Cl, double h, double tau, double Gam_g, double alpha, double beta, double eps, int k, int l, int g)

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
						nu_temp = h * nu_temp / (dist * dist + eps * h * h);
						F_nu = (-alpha * Cnu * nu_temp + beta * nu_temp * nu_temp) / (0.5 * (rho_gas[i] + rho_gas[j]));
					}

					//	F_tens = kapp * mas[j] * dist * W(dist, h0);
						//e_temp = e_temp + (mas_gas[i] * p_gas[i] / (rho_gas[i] * rho_gas[i]) + mas_gas[j] * p_gas[j] / (rho_gas[j] * rho_gas[j]) + mas_gas[j] / 2.0 * F_nu) * 
						//	((Vx[i] - Vx[j]) * dW(x[i] - x[j], h) + (Vy[i] - Vy[j]) * dW(y[i] - y[j], h) + (Vz[i] - Vz[j]) * dW(z[i] - z[j], h));
						// e_temp = e_temp + (mas_gas[i] * p_gas[i] / (rho_gas[i] * rho_gas[i]) + mas_gas[j] * p_gas[j] / (rho_gas[j] * rho_gas[j]) + mas_gas[j] / 2.0 * F_nu) *
						//	((Vx[i] - Vx[j]) * dW(x[i] - x[j], h) + (Vy[i] - Vy[j]) * dW(y[i] - y[j], h) + (Vz[i] - Vz[j]) * dW(z[i] - z[j], h));
					e_temp[i] = e_temp[i] + (mas_gas[i] * p_gas[i] / (rho_gas[i] * rho_gas[i]) + mas_gas[i] * F_nu / 2.0) *
						(nu1x + nu1y + nu1z) * dW(dist, h) / dist;
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



__global__ void V_dustKernel_IDIC(double* x, double* y, double* z, double* Vx_dust, double* Vy_dust, double* Vz_dust,  double* Vx_g_average, double* Vy_g_average, double* Vz_g_average, double* t_stop_average, int* Ind, 
	double X_min, double Y_min, double Z_min, double  average_cell_width, int  Number_of_average_cell_x, int  Number_of_average_cell_y, int Number_of_average_cell, double tau, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	int cell_num;

	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		cell_num = int((x[i] - X_min) / average_cell_width) + (int((y[i] - Y_min) / average_cell_width)) * Number_of_average_cell_x + (int((z[i] - Z_min) / average_cell_width)) * Number_of_average_cell_x * Number_of_average_cell_y; 
		Vx_dust[i] = (Vx_dust[i] / tau + Vx_g_average[cell_num] / t_stop_average[cell_num]) / (1.0 / tau + 1.0 / t_stop_average[cell_num]);
		Vy_dust[i] = (Vy_dust[i] / tau + Vy_g_average[cell_num] / t_stop_average[cell_num]) / (1.0 / tau + 1.0 / t_stop_average[cell_num]);
		Vz_dust[i] = (Vz_dust[i] / tau + Vz_g_average[cell_num] / t_stop_average[cell_num]) / (1.0 / tau + 1.0 / t_stop_average[cell_num]);
		
		

	}

}


__global__ void V_gasKernel_IDIC(double* x, double* y, double* z, double* Vx_gas, double* Vy_gas, double* Vz_gas, double* Vx_d_average_1, double* Vy_d_average_1, double* Vz_d_average_1, double* Vx_d_average_2, double* Vy_d_average_2, double* Vz_d_average_2, 
	double* Psi_av_new_x, double* Psi_av_new_y, double* Psi_av_new_z, double* t_stop_average_1, double* t_stop_average_2, double *eps_cell_1, double *eps_cell_2, int* Ind, double X_min, double Y_min, double Z_min, double  average_cell_width, int  Number_of_average_cell_x, int  Number_of_average_cell_y, int Number_of_average_cell, double tau, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	double tmp1, tmp_x, tmp_y, tmp_z;
	int cell_num;

	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		cell_num = int((x[i] - X_min) / average_cell_width) + (int((y[i] - Y_min) / average_cell_width)) * Number_of_average_cell_x + (int((z[i] - Z_min) / average_cell_width)) * Number_of_average_cell_x * Number_of_average_cell_y;

		tmp1 = 0.0;
		tmp_x = 0.0;
		tmp_y = 0.0;
		tmp_z = 0.0;

		tmp1 = tmp1 + eps_cell_1[cell_num] / t_stop_average_1[cell_num] + eps_cell_2[cell_num] / t_stop_average_2[cell_num];
		tmp_x = tmp_x + eps_cell_1[cell_num] * Vx_d_average_1[cell_num] / t_stop_average_1[cell_num] + eps_cell_2[cell_num] * Vx_d_average_2[cell_num] / t_stop_average_2[cell_num];
		tmp_y = tmp_y + eps_cell_1[cell_num] * Vy_d_average_1[cell_num] / t_stop_average_1[cell_num] + eps_cell_2[cell_num] * Vy_d_average_2[cell_num] / t_stop_average_2[cell_num];
		tmp_z = tmp_z + eps_cell_1[cell_num] * Vz_d_average_1[cell_num] / t_stop_average_1[cell_num] + eps_cell_2[cell_num] * Vz_d_average_2[cell_num] / t_stop_average_2[cell_num];

		Vx_gas[i] = (Vx_gas[i] / tau + tmp_x + Psi_av_new_x[cell_num]) / (1.0 / tau + tmp1);
		Vy_gas[i] = (Vy_gas[i] / tau + tmp_y + Psi_av_new_y[cell_num]) / (1.0 / tau + tmp1);
		Vz_gas[i] = (Vz_gas[i] / tau + tmp_z + Psi_av_new_z[cell_num]) / (1.0 / tau + tmp1);
		}
}



__global__ void MoveKernel_IDIC(double* x, double* y, double* z, double* Vx, double* Vy, double* Vz, int* Ind, double tau, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;

	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		x[i] = x[i] + Vx[i] * tau;
		y[i] = y[i] + Vy[i] * tau;
		z[i] = z[i] + Vz[i] * tau;
	}

}

__global__ void EnergyKernel_IDIC(double* e_gas, double* e_temp, int* Ind, double tau, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;

	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		e_gas[i] = e_gas[i] + e_temp[i] * tau;
	}

}