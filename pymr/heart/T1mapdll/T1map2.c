#define EXPORT __declspec(dllexport)
#include "lmcurve.h"
#include <stdio.h>
#include <math.h>
/* model function: a parabola */

double f( double t, const double *p )
{
	return 	fabs(p[0]-p[1]*1.0*exp(-t/p[2]));
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
	double guess[8] ={100,200,500,800,1000,1500,2000,2500};
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