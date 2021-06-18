'''
This is an object to take care of unit vector
'''
from xarrayuvecs.uniform_dist import unidist
import xarrayuvecs.lut2d as lut2d

import datetime
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from sklearn.neighbors import KernelDensity
import scipy

@xr.register_dataarray_accessor("uvecs")

class uvecs(object):
    '''
    This is a classe to work on unit vector in xarray environnement that respect the -1 symmetry (i.e. u is equivalent to -u)
    
    .. note:: xarray does not support heritage from xr.DataArray may be the day it support it, we could move to it
    '''
    
    def __init__(self, xarray_obj):
        '''
        Constructor for uvec. The univ vector u should be pass as azimuth and colatitude in radian
        Colatitude : angle between u-vector and z vector [0 pi/2]
        Azimuth : angle between the projection of u-vector in xOy plan and x-vector [0 2pi]
        
        :param xarray_obj: dimention should be (n,m,2), xarray_obj[n,m,0]=azimuth , xarray_obj[n,m,1]=colatitude
        :type xarray_obj: xr.DataArray
        '''
        self._obj = xarray_obj 
    pass
    

#-----------------------------vector representation-------------------------------------        
    def azi_col(self):
        '''
        :return out: the azimuth out[n,m,0] and colatitude out[n,m,0], dim (n,m,2)
        :rtype out: np.array
        '''
        return np.moveaxis(np.array([self._obj[:,:,0],self._obj[:,:,1]]),0,-1)
        
    def bunge_euler(self):
        '''
        This is from the Euler angle, Bunge convention
        1. rotate around z-axis of phi1
        2. rotate around x'-axis of phi
        :return out: phi1 and phi, out[n,m,0]=phi1, out[n,m,1]=phi
        :rtype out: np.array
        '''
        BE=np.moveaxis(np.array([np.mod(self._obj[:,:,0]+np.pi/2.,2*np.pi),self._obj[:,:,1]]),0,-1)
        
        return xr.DataArray(BE,dims=[self._obj.coords.dims[0],self._obj.coords.dims[1],'vbe'])


    def xyz(self):
        '''
        Return axis in cartesian coordinate
        :return out: out[n,m,0]=x, out[n,m,1]=u , out[n,m,2]=z
        :rtype out: np.array
        '''
        XYZ=np.moveaxis(np.array([np.cos(self._obj[:,:,0])*np.sin(self._obj[:,:,1]),np.sin(self._obj[:,:,0])*np.sin(self._obj[:,:,1]),np.cos(self._obj[:,:,1])]),0,-1)

        return xr.DataArray(XYZ,dims=[self._obj.coords.dims[0],self._obj.coords.dims[1],'vc'])
    
#-----------------------------------colormap function-------------------------------------
    def calc_colormap(self,**kwargs):
        '''
        Compute the colormap value
        :param nlut: size of the lut (default:512)
        :type nlut: int
        :param semi: colorbar option
        '''
        rlut=lut2d.lut(circle=False,**kwargs)
        nlut=np.shape(rlut)[0]
        
        XX=np.int32((nlut-1)/2*np.multiply(np.sin(self._obj[:,:,1]),-np.sin(self._obj[:,:,0]))+(nlut-1)/2)
        YY=np.int32((nlut-1)/2*np.multiply(np.sin(self._obj[:,:,1]),np.cos(self._obj[:,:,0]))+(nlut-1)/2)
        
        id=XX<0
        XX[id]=0
        YY[id]=0
        
        idx,idy=np.where(id==True)
        img=rlut[XX,YY]
        img[idx,idy,:]=np.array([255,255,255])
        return xr.DataArray(img,dims=[self._obj.coords.dims[0],self._obj.coords.dims[1],'img'])
#--------------------------------------------------------------------------------------------
    def OT2nd(self):
        '''
        Compute the second order orientation tensor
        
        :return eigvalue: eigen value w[i]
        :rtype eigvalue: np.array
        :return eigvector: eigen vector v[:,i]
        :rtype eigvector: np.array
        
        .. note:: eigen value w[i] is associate to eigen vector v[:,i] 
        '''
        u_xyz=self.xyz()
        ux=np.concatenate([np.array(u_xyz[:,:,0]).flatten(),-np.array(u_xyz[:,:,0]).flatten()])
        uy=np.concatenate([np.array(u_xyz[:,:,1]).flatten(),-np.array(u_xyz[:,:,1]).flatten()])
        uz=np.concatenate([np.array(u_xyz[:,:,2]).flatten(),-np.array(u_xyz[:,:,2]).flatten()])
        
        
        a11 = np.float32(np.nanmean(np.float128(np.multiply(ux,ux))))
        a22 = np.float32(np.nanmean(np.float128(np.multiply(uy,uy))))
        a33 = np.float32(np.nanmean(np.float128(np.multiply(uz,uz))))
        a12 = np.float32(np.nanmean(np.float128(np.multiply(ux,uy))))
        a13 = np.float32(np.nanmean(np.float128(np.multiply(ux,uz))))
        a23 = np.float32(np.nanmean(np.float128(np.multiply(uy,uz))))
        
        Tensor=np.array([[a11, a12, a13],[a12, a22, a23],[a13, a23, a33]])
        eigvalue,eigvector=np.linalg.eig(Tensor)
        
        idx = eigvalue.argsort()[::-1]
           
        return eigvalue[idx],eigvector[:,idx]
#--------------------------------------------------------------------------------------------
    def misorientation_profile(self,xx,yy,degre=True,method="nearest",**kwargs):
        '''
        Extract value for xx,yy
        :param xx: x coordinate
        :type xx: np.array()
        :param yy: y coordinate
        :type yy: np.array()
        :param degre: Do you want the angle in degree (default:True)
        :type degre: bool
        :param **kwargs: xr.sel
        '''
        ori=self._obj.sel(x=xx,y=yy, method=method,**kwargs)
        # trasform it in numpy array in cartesien coordinate
        vxyz=np.array(ori.uvecs.xyz())[0]
        # compute the misorientation to origin
        mis2o=np.arccos(np.round(np.dot(vxyz,vxyz[0]),10))
    
        mis2o[np.where(mis2o>np.pi/2)]=np.pi-mis2o[np.where(mis2o>np.pi/2)]
        # compute misorientation from previous
        vxyzm=np.zeros(np.shape(vxyz))
        vxyzm[:,:]=np.nan
        vxyzm[1::,:]=vxyz[0:-1,:]
        mis2p=np.arccos(np.round(np.diag(np.dot(vxyz,np.transpose(vxyzm))),10))
        mis2p[np.where(mis2p>np.pi/2)]=np.pi-mis2p[np.where(mis2p>np.pi/2)]
        #compute distance
        d=((xx-xx[0])**2+(yy-yy[0])**2)**0.5
        
        if degre:
            coeff=180/np.pi
        else:
            coeff=1.
        
        ds=xr.Dataset(
        {
            'mis2i': (['d'],mis2o*coeff),
            'mis2p': (['d'],mis2p*coeff),
        },
        coords={'d':d})
        return ds
    
    
#--------------------------------------------------------------------------------------------
    def mis_angle(self,random=False):
        '''
        Compute the misorientation with the neighbouring grain
        :param random: suffle the image and compute the angle
        :type random: bool

        '''
        phi1=np.array(self.bunge_euler())[:,:,0]
        phi=np.array(self.bunge_euler())[:,:,1]

        if random:
            np.random.shuffle(phi1)
            np.random.shuffle(phi)
            phi1=phi1.flatten()
            phi=phi.flatten()
            phi1 = phi1[~np.isnan(phi1)]
            phi = phi[~np.isnan(phi)]
            dd=np.int(np.sqrt(len(phi1)))
            phi1=phi1[0:dd**2].reshape([dd,dd])
            phi=phi[0:dd**2].reshape([dd,dd])

        mat=np.zeros([3,3])
        mat[0,1]=1
        phi_a=scipy.signal.convolve2d(phi,mat,mode='same',boundary='symm')
        phi1_a=scipy.signal.convolve2d(phi1,mat,mode='same',boundary='symm')

        mat=np.zeros([3,3])
        mat[1,0]=1
        phi_b=scipy.signal.convolve2d(phi,mat,mode='same',boundary='symm')
        phi1_b=scipy.signal.convolve2d(phi1,mat,mode='same',boundary='symm')


        mat=np.zeros([3,3])
        mat[1,2]=1
        phi_c=scipy.signal.convolve2d(phi,mat,mode='same',boundary='symm')
        phi1_c=scipy.signal.convolve2d(phi1,mat,mode='same',boundary='symm')


        mat=np.zeros([3,3])
        mat[2,1]=1
        phi_d=scipy.signal.convolve2d(phi,mat,mode='same',boundary='symm')
        phi1_d=scipy.signal.convolve2d(phi1,mat,mode='same',boundary='symm')

        phi1_s=[phi1_a,phi1_b,phi1_c,phi1_d]
        phi_s=[phi_a,phi_b,phi_c,phi_d]
        if random:
            tot=np.zeros([dd,dd,4])
        else:
            tot=np.zeros([len(self._obj.y),len(self._obj.x),4])
        
        for i in range(4):
            nphi1=phi1_s[i]
            nphi=phi_s[i]
            res=np.arccos(np.round(np.sin(phi1)*np.sin(nphi1)*np.sin(phi)*np.sin(nphi)+np.cos(phi1)*np.cos(nphi1)*np.sin(phi)*np.sin(nphi)+np.cos(phi)*np.cos(nphi),5))
            #put everything between 0  and pi/2 because c=-c
            id=np.where(res>np.pi/2)
            res[id]=np.pi-res[id] 

            res[0,:] = np.nan  # delete first row 
            res[-1,:] = np.nan
            res[:,0] = np.nan
            res[:,-1] = np.nan
            tot[:,:,i]=res
            

        if random:
            tot=tot.flatten()
            tot=tot[~np.isnan(tot)]
        else:
            tot=xr.DataArray(tot,dims=[self._obj.coords.dims[0],self._obj.coords.dims[1],'misAngle'])

        return tot
#--------------------------------------------------------------------------------------------
    def plotODF(self,nbr=10000,bw=0.2,projz=1,plotOT=True,angle=np.array([30.,60.]),cline=10,**kwargs):
        
        #compute phi theta under the nice form for kde fit
        u_xyz=self.xyz()
        
        ux=np.concatenate([np.array(u_xyz[:,:,0]).flatten(),-np.array(u_xyz[:,:,0]).flatten()])
        uy=np.concatenate([np.array(u_xyz[:,:,1]).flatten(),-np.array(u_xyz[:,:,1]).flatten()])
        uz=np.concatenate([np.array(u_xyz[:,:,2]).flatten(),-np.array(u_xyz[:,:,2]).flatten()])
        
        ux=ux[~np.isnan(ux)]
        uy=uy[~np.isnan(uy)]
        uz=uz[~np.isnan(uz)]
        
        
        if nbr!=0:
            if nbr>len(ux):
                nbr=len(ux)
                
            rng = np.random.default_rng()
            numbers = rng.choice(len(ux), size=nbr, replace=False)
            
            ux=ux[numbers]
            uy=uy[numbers]
            uz=uz[numbers]
                        
        
        phi=np.arccos(uz)-np.pi/2.
        theta=np.arctan2(uy,ux)-np.pi
        
        
        #compite the kde
        kde = KernelDensity(bandwidth=bw, metric='haversine',kernel='gaussian', algorithm='ball_tree')
        kde.fit(np.transpose(np.array([phi,theta])))
        
        # Prepare the plot
        val=unidist
        dim=int(np.size(val)/3)
        vs=val.reshape([dim,3])
        id=np.where(vs[:,2]>0)
        vs_u=vs[id[0],:]
        
        # add point on the disc for contourf
        tot=10000
        omega = np.linspace(0, 2*np.pi, tot)
        zcir = np.zeros(tot)
        xcir = np.cos(omega)
        ycir = np.sin(omega)
        
        vs_x=np.concatenate([vs[:,0],xcir])
        vs_y=np.concatenate([vs[:,1],ycir])
        vs_z=np.concatenate([vs[:,2],zcir])
        
        id=np.where(vs_z<0)
        vs_x[id]=-vs_x[id]
        vs_y[id]=-vs_y[id]
        vs_z[id]=-vs_z[id]
        
        phi_e=np.arccos(vs_z)
        theta_e=np.arctan2(vs_y,vs_x)
        
        weights=kde.score_samples(np.transpose(np.array([phi_e-np.pi/2.,theta_e-np.pi])))
        
        # Choose the type of projection
        if projz==0:
            LpL=1./(1.+vs_z)
            xx=LpL*vs_x
            yy=LpL*vs_y
            rci=np.multiply(1./(1.+np.sin((90-angle)*np.pi/180.)),np.cos((90-angle)*np.pi/180.))
            rco=1.
        else:
            xx = np.multiply(2*np.sin(phi_e/2),np.cos(theta_e))
            yy = np.multiply(2*np.sin(phi_e/2),np.sin(theta_e))
            rci=2.*np.sin(angle/2.*np.pi/180.)
            rco=2.**0.5
            
        # plot contourf
        triang = tri.Triangulation(xx, yy)
        plt.tricontour(xx, yy, np.exp(weights), cline, linewidths=0.5, colors='k')
        plt.tricontourf(xx, yy, np.exp(weights), cline, **kwargs)
        
        
        plt.colorbar(orientation='vertical',aspect=4,shrink=0.5)
        # Compute the outer circle
        omega = np.linspace(0, 2*np.pi, 1000)
        x_circle = rco*np.cos(omega)
        y_circle = rco*np.sin(omega)
        plt.plot(x_circle, y_circle,'k', linewidth=3)
        # compute a 3 circle
        if np.size(angle)>1:
            for i in list(range(len(rci))):
                x_circle = rci[i]*np.cos(omega)
                y_circle = rci[i]*np.cos(i*np.pi/180.)*np.sin(omega)
                
                plt.plot(x_circle, y_circle,'k', linewidth=1.5)
                plt.text(x_circle[200], y_circle[300]+0.04,'$\phi$='+str(angle[i])+'°')
            # plot Theta line
            plt.plot([0,0],[-1*rco,1*rco],'k', linewidth=1.5)
            plt.text(rco-0.2, 0+0.06,'$\Theta$=0°')
            plt.text(-rco+0.1, 0-0.06,'$\Theta$=180°')
            plt.plot([-rco,rco],[0,0],'k', linewidth=1.5)
            plt.text(-0.25, rco-0.25,'$\Theta$=90°')
            plt.text(0.01, -rco+0.15,'$\Theta$=270°')
            plt.plot([-0.7071*rco,0.7071*rco],[-0.7071*rco,0.7071*rco],'k', linewidth=1.5)
            plt.plot([-0.7071*rco,0.7071*rco],[0.7071*rco,-0.7071*rco],'k', linewidth=1.5)
          
            
        # draw a cross for x and y direction
        plt.plot([1*rco, 0],[0, 1*rco],'+k',markersize=12)
        # write axis
        plt.text(1.05*rco, 0, r'X')
        plt.text(0, 1.05*rco, r'Y')
        plt.axis('equal')
        plt.axis('off')
                   
        
        if plotOT:
            eigvalue,eigvector=self.OT2nd()
            for i in list(range(3)): # Loop on the 3 eigenvalue
                if (eigvector[2,i]<0):
                    v=-eigvector[:,i]
                else:
                    v=eigvector[:,i]
                    
                    
                if projz==0:    
                    LpLv=1./(1.+v[2])
                    xxv=LpLv*v[0]
                    yyv=LpLv*v[1]
                else:
                    phiee=np.arccos(v[2])
                    thetaee=np.arctan2(v[1],v[0])
                    xxv = np.multiply(2*np.sin(phiee/2),np.cos(thetaee))
                    yyv = np.multiply(2*np.sin(phiee/2),np.sin(thetaee))
                    
                plt.plot(xxv,yyv,'sk',markersize=8)
                plt.text(xxv+0.04, yyv+0.04,str(round(eigvalue[i],2)))
            
#-------------------------------------------------------------------------------------------            
    def calc_schmid(self,axis):
        '''
        Compute the schimd sc=abs(cos(a)sin(a)) where a is the angle between c and axis
        :param axis: axis from which you want to compute the schimd factor in cartesien coordinate (X,Y,Z)
        :type axis: np.array
        '''

        ori=self.xyz()
        angle=np.arccos(np.sum(ori*axis,axis=-1))
        schmid=np.abs(np.cos(angle)*np.sin(angle))

        return schmid
    
#--------------------------------------------------------------------------------------------
    def inner_angle(self, other):
        '''
        Compute the inner angle between to uvecs DataArray
        :param other: DataArray of same dimention than self and compatible with uvec
        :param other: xr.DataArray
        :return:
        :rtype: xr.DataArray
        '''
        o1=self.xyz()
        o2=other.uvecs.xyz()
        angle=np.array(np.arccos(np.sum(o1*o2,axis=-1)))
        id=np.where(angle>np.pi/2)
        angle[id]=np.pi-angle[id]
        
        return xr.DataArray(angle,dims=self._obj.coords.dims[0:2])
