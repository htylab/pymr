#define EXPORT __declspec(dllexport)
#include "lmcurve.h"
#include <stdio.h>
#include <math.h>
/* model function: a parabola */

double f( double t, const double *p )
{
	return 	fabs(p[0]-p[1]*1.0*exp(-t/p[2]));
}

double f2( double t, const double *p )
{
	return 	(p[0]-p[1]*1.0*exp(-t/p[2]));
}
double t1value(double T1,double A,double B,double tx)
{
	return abs(A-B*exp(-tx/T1));
}
//EXPORT void __stdcall syn(double *t,int t_num,double *y,long y_num,double *col) 
EXPORT void __stdcall syn(double *t,int t_num,double *y,long y_num, double SI_th,double *T1map, double *Amap, double *Bmap,double *T1starmap, double *Errmap) 

{
    int n = 3;
    double par[3] = { 100, 200, 2500 }; 
    double par_init[3] = { 100, 200, 2500 }; 

	//double guess[8] ={100,200,500,800,1000,1500,2000,2500};
	double guess[8] ={200,500,800,1100,1400,1700,2000};
	//double guess[8] ={1000,1000,1000,1000,1000,1000,1000,1000};
	double A[8]={0,0,0,0,0,0,0,0};
	double B[8]={0,0,0,0,0,0,0,0};
	double T1val[8]={0,0,0,0,0,0,0,0};
	double T1starval[8]={0,0,0,0,0,0,0,0};
	double err[8]={0,0,0,0,0,0,0,0};
	double errd;
    //int m_dat = t_num;
	int jj,ii,kk;
	//double tempvalue;
	double maxTI,minErr;
	double SI_maxTI;
    unsigned int argmaxTI,argminErr;
    double tempy[100];
	double miny;
    unsigned int argminy;
	//double tx;
    lm_control_struct control = lm_control_double;
    lm_status_struct status;
    control.verbosity = 0;
    //find maximum TI to initialize par
    maxTI = t[0];
    argmaxTI = 0;
    for (ii = 1; ii < t_num; ii++)
    {
        if (t[ii] > maxTI)
        {
        	maxTI = t[ii];
        	argmaxTI = ii;
        }
    }
	
    for (ii=0; ii<y_num;ii++) 
	{
		SI_maxTI = *(y+ii*t_num+argmaxTI);

		if (SI_maxTI >SI_th)
        {
            for (jj=0; jj<8; jj++) {
            	par[0]=SI_maxTI;
                par[1]=par[0]*2;
            	par[2]=guess[jj];
            	//par[2]=1000;
            	lmcurve( n, par, t_num, t, y+ii*t_num, f, &control, &status );
            	A[jj]=par[0] ;//A
            	B[jj]=par[1] ;//B
            	T1starval[jj]=par[2];
            	T1val[jj]=(par[1]/par[0]-1)*par[2]; //T1* to T1 
            	errd =0;
            	for (kk=0;kk<t_num;kk++){
            		errd+=fabs(y[kk+ii*t_num]-f(t[kk],par));
            	    }			
            	err[jj] =errd;
            }
            minErr=err[0];
	        argminErr=0;	
	        for (jj=1; jj<8; jj++) {
		        if (err[jj] < minErr ) {
			        minErr = err[jj];
			        argminErr = jj;
			    }
	        }
	        T1map[ii] = T1val[argminErr];
	        Amap[ii] = A[argminErr];
	        Bmap[ii] = B[argminErr];
	        T1starmap[ii] = T1starval[argminErr];
	        Errmap[ii] = minErr;
	        par_init[0] = Amap[ii];
	        par_init[1] = Bmap[ii];
	        par_init[2] = T1starmap[ii];

	                    // step 2. 假設最小值兩側為最可能的極小值曲域
            minErr=1e9;
            for (jj=0;jj<3;jj++){
            	for (kk=0;kk<t_num;kk++){
                	tempy[kk]= y[ii*t_num+kk];
                	if (kk <= argminy +jj - 1) 
            		    tempy[kk]= tempy[kk] * -1;
                }
                par[0]=par_init[0];
                par[1]=par_init[1];
                par[2]=par_init[2];
                lmcurve( n, par, t_num, t, tempy, f2, &control, &status );
                errd =0;
                for (kk=0;kk<t_num;kk++)
            	    errd+=fabs(tempy[kk]-f2(t[kk],par));

                if (errd < minErr){
            	    minErr = errd;
                    T1map[ii] = (par[1]/par[0]-1)*par[2];
                    //T1map[ii] = T1val[argminErr];
	                Amap[ii] = par[0];
	                Bmap[ii] = par[1];
	                T1starmap[ii] = par[2];
	                Errmap[ii] = minErr;
	            }
            }
	    }
        else{	
	        T1map[ii] = 0;
	        Amap[ii] = 0;
	        Bmap[ii] = 0;
	        T1starmap[ii] = 0;
	        Errmap[ii] = 0;
        	}

	}

}