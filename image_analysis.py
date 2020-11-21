import matplotlib.pyplot as plt
import imageio as img
import numpy as np
import scipy as sp
import skimage as ski #for different image analysis algorithms
import skimage.filters

def psf_airy(scale_true,lam,NA=1.3,size=None):
    #NA = 1.3 by default - numerical aperture of light microscope
    r_airy = 1.22*lam/(2*NA)#radius of first minima
    
    if size is None: size = np.ceil((3.5*r_airy/scale_true)/2.)*2 + 1
    #size is taken to be 3.5x the first minima round up to odd integer
    #size can also be specified by the user
    
    a = 3.83/(r_airy/scale_true)*((size-1)/2) #in terms of rho
    res = a/((size-1)/2)
    x = np.arange(-a, a+res, res)
    y = np.arange(-a, a+res, res)
    X, Y = np.meshgrid(x, y)
    Rh = np.sqrt(X**2+Y**2)
    z = (2*sp.special.j1(Rh)/(Rh))**2
    return z

def simulator(pic,scale_true,lam,ccd_scale,SNr,bkg,NA = 1.3,psf='gaussian',size=None,m=None,n=None):
    #convolving with psf
    if psf=='gaussian': convolved = ski.filters.gaussian(pic,psf_sig(lam,scale_true,NA=NA))
    elif psf == 'airy': 
        kernel = psf_airy(scale_true,lam,NA=NA,size=size)
        convolved = sp.ndimage.convolve(pic,kernel, mode = 'reflect')
    
    #pixelation
    blck_siz = ([np.round(ccd_scale/scale_true).astype('int')]*2); blck_siz.extend([1]*(len(np.shape(convolved))-2))
    #the block_reduce function cannot deal with stacks of images, the appropriate block size is specified 
    #based on the size of the stack 
    pixelated = ski.measure.block_reduce(convolved, block_size = tuple(blck_siz), 
                                         func = np.sum, cval = 0.0)
    
    #removing overhangs:
    #the `block reduce` function pads 0 values if the block is not exactly divisible by the image
    if (np.shape(pixelated)[0]*blck_siz[0]>np.shape(convolved)[0]): m = -1
    if (np.shape(pixelated)[1]*blck_siz[1]>np.shape(convolved)[1]): n = -1
    pixelated = pixelated[:m,:n]
    
    #noise addition
    S = SNr**2
    if np.max(pixelated)>0: noisy_sgnl = np.random.poisson(pixelated*S/np.max(pixelated))
    else: noisy_sgnl = pixelated
    noisy_bkg = np.random.poisson(bkg*np.ones(np.shape(noisy_sgnl)))
    final_img = noisy_bkg+noisy_sgnl
    return final_img

def psf_sig(lam,scale_true,NA=1.3):
    psf_sigma = 0.45*lam/(2*NA)/scale_true 
    #this is the std derived by fitting gaussian to airy function
    return psf_sigma

def crlb(img,r,c,scale_true,lam,ccd_scale,SNr,bkg,NA = 1.3,N=200,psf = 'gaussian'):
    #display the point for which the crlb is being calculated
    plt.imshow(img, cmap = 'gray')
    plt.title('Point for calculating CRLB')
    plt.plot(c,r,'o', color = 'crimson', markersize = 5, alpha = 0.5); 
    #row column according to plt.plot convention is opposite
    
    #transform point to the CCD image coordinates
    ccd_arr_siz = np.round((scale_true/ccd_scale)*np.array(np.shape(img)))
    r_ccd = np.round((ccd_arr_siz[0])/(np.shape(img)[0])*(r+0.5)-0.5).astype('int')
    c_ccd = np.round((ccd_arr_siz[1])/(np.shape(img)[1])*(c+0.5)-0.5).astype('int')
    
    #stack the true image N times
    stack = np.zeros((np.shape(img)[0],np.shape(img)[1],N))
    for i in range(np.shape(stack)[2]):
        stack[:,:,i] = img
    
    #simulate the stack
    stack_sim = simulator(stack,scale_true,lam,ccd_scale,SNr,bkg,psf = psf)
    
    #CRLB calculation
    d2logim = np.log(stack_sim[r_ccd,c_ccd+1,:]) + np.log(stack_sim[r_ccd,c_ccd-1,:]) - 2*np.log(stack_sim[r_ccd,c_ccd,:])   
    N_pix = np.mean(np.sum(np.sum(stack_sim,0),0)) - (bkg*ccd_arr_siz[0]*ccd_arr_siz[1])    
    crlb = (1/np.sqrt(-np.mean(d2logim))/np.sqrt(N_pix))*np.sqrt(2)*ccd_scale
    
    #minimum possible CRLB
    crlb_min = np.sqrt(2)*lam/(2*np.pi*NA*np.sqrt(N_pix))
    return crlb, crlb_min

def COM(pic):
    if np.shape(np.shape(pic))[0]<2: pic = np.array([pic])
    x_com = np.zeros(np.shape(pic)[0])
    y_com = np.zeros(np.shape(pic)[1])
    for i in np.arange(0, np.max(np.shape(pic))):
        if (i<np.shape(pic)[1]): x_com += ((i)*pic[:,i]) #Xcom along columns
        if (i<np.shape(pic)[0]): y_com += ((i)*pic[i,:]) #Ycom along rows
    x_com = np.sum(x_com)/np.sum(pic) #normalize
    y_com = np.sum(y_com)/np.sum(pic) #normalize
    return x_com, y_com

def transform_ccd_coords(img,scale_true,ccd_scale,r,c):
    #finding exact ccd_arr_siz
    ccd_arr_siz = (scale_true/ccd_scale)*np.array(np.shape(img))   
    r_ccd = (ccd_arr_siz[0])/(np.shape(img)[0])*(r+0.50)-0.50 #appropriate shifts according to image grid
    c_ccd = (ccd_arr_siz[1])/(np.shape(img)[1])*(c+0.50)-0.50
    return r_ccd,c_ccd

def gaussian2d_symm(mesh, A, x0, y0, sigma, offset, ravel = True):
    #symmetric gaussian for x,y coordinates with 5 parameters A-height, (x0,y0)-centre
    #sigma - std dev and offset - baseline
    px = mesh[0]; py = mesh[1]
    if ravel == True: dr = np.vstack([px.ravel() - x0, py.ravel() - y0])
    else: dr = np.array([px - x0, py - y0])
    gauss_dist = A*np.exp(-np.sum(dr*dr,0)/(2*(sigma**2))) + offset
    return gauss_dist

def gaussian2d_asymm(mesh, A, x0, y0, sigma1, sigma2, offset, theta, ravel = True):    
    #asymmetric gaussian for x,y coordinates with 6 parameters A-height, (x0,y0)-centre
    #sigma1 - std dev along 1st axis, sigma2 - std dev along 2nd axis, offset - baseline, theta - rotation
    
    px = mesh[0]; py = mesh[1]

    a = ((np.cos(theta)**2) / (2*sigma1**2)) + ((np.sin(theta)**2) / (2*sigma2**2))
    b = -((np.sin(2*theta)) / (4*sigma1**2)) + ((np.sin(2*theta)) / (4*sigma2**2))
    c = ((np.sin(theta)**2) / (2*sigma1**2)) + ((np.cos(theta)**2) / (2*sigma2**2))
    
    if ravel == True: dr2 = np.vstack([a*(px.ravel() - x0)**2, 2*b*(px.ravel() - x0)*(py.ravel() - y0), c*(py.ravel() - y0)**2])
    else: dr2 = np.array([a*(px - x0)**2, 2*b*(px - x0)*(py - y0), c*(py - y0)**2])
    gauss_dist = A*np.exp(-np.sum(dr2,0)) + offset
    return gauss_dist

def gauss2_fit_MLE(pic, params0 = None, pic_off = 10.0, tolz = 1e-6, symm = True, n=0, det_lim = 1.0, suppress = False):
    #if symm is false, bivariate gaussian with rotation is fit
    
    #initial values
    A_ini = np.max(pic)-np.min(pic); offs_ini = np.min(pic); sig_ini = np.min(np.shape(pic))/4.0
    
        #find brightest pixels: (if >1, choose one)
        #det_lim decides what fraction of the brightest pixels you sample from: for instance det_lim = 0.95 implies
        #you choose one pixel from all the pixels that have a brightness value >95% of the brightest pixel
    x0_ini = int(np.where(pic >= det_lim*np.max(pic))[1][n]); y0_ini = int(np.where(pic >= det_lim*np.max(pic))[0][n])
    theta_ini = 0.0
    
    px, py = np.meshgrid(np.arange(0,np.shape(pic)[1]),np.arange(0,np.shape(pic)[0]))
    
    #image offset
    pic = pic+pic_off #prevents numerical error
    offs_ini = offs_ini+pic_off
    
    #set initial parameters
    if params0 is None: 
        if symm == True: params0 = np.array([A_ini, x0_ini, y0_ini, sig_ini, offs_ini])
        else: params0 = np.array([A_ini, x0_ini, y0_ini, sig_ini, sig_ini, offs_ini, theta_ini])
    
    #specify limits (x0, y0 should not be out of the image)
    if symm == True: bnds = ((None,None),(0,np.shape(pic)[1]),(0,np.shape(pic)[0]),(None,None),(0,None))
    else: bnds = ((None,None),(0,np.shape(pic)[1]),(0,np.shape(pic)[0]),(None,None),(None,None), (0,None), (None,None))
     
    #minimize -ve log of likelihood with scipy.optimize.minimize
    result = sp.optimize.minimize(negL_gauss, params0, args = (px,py,pic,symm), method = 'L-BFGS-B', 
                                  bounds = bnds, options = {'gtol':tolz})
    
    #prints whether tolerance limit was reached or not
    if suppress == False: print('Sucess:', result.success)
    
    #get final parameters
    A = result.x[0]; x0 = result.x[1]; y0 = result.x[2]
    
    if symm == False:
        sigma1 = abs(result.x[3]); sigma2 = abs(result.x[4]) #sigma should be +ve value
        sigma = np.array([sigma1,sigma2])
        offset = result.x[5] - pic_off
        theta = result.x[6]
    else:
        sigma = abs(result.x[3]); offset = result.x[4] - pic_off; theta = None
    
    return A,x0,y0,sigma,offset,theta

def negL_gauss(params,px,py,pic,symm = True):
    if symm == True: gaussprob = gaussian2d_symm((px,py),params[0],params[1],params[2],params[3],params[4],ravel = True)
    else: gaussprob = gaussian2d_asymm((px,py),params[0],params[1],params[2],
                                       params[3],params[4],params[5],params[6],ravel = True)
    
    #negative log of likelihood for spatial gaussian with poisson intensity noise
    Lk = (pic.ravel()*np.log(gaussprob) - gaussprob)
    negL = -np.sum(Lk.ravel())
    return negL

def gauss2_fit_LS(data):
    #initial values
    A_ini = np.max(data)-np.min(data); offs_ini = np.min(data); sig_ini = np.min(np.shape(data))/4.0
        #find brightest pixels:
    x0_ini = int(np.where(data == np.max(data))[1][0]); y0_ini = int(np.where(data == np.max(data))[0][0])
    px, py = np.meshgrid(np.arange(0,np.shape(data)[1]),np.arange(0,np.shape(data)[0]))

    params0 = np.array([A_ini,x0_ini,y0_ini,sig_ini,offs_ini])

    data = data.ravel()
    popt,_ = sp.optimize.curve_fit(gaussian2d_symm, (px,py), data, p0 = params0)
    A = popt[0]; x0 = popt[1]; y0 = popt[2]; sig = popt[3]; offset = popt[4]
    return A, x0, y0, sig, offset

def local_maxima(pic,grid_size = None,plot = False,color = 'thistle'):
    #set grid size is 1/6th of the smaller axis
    if grid_size is None: grid_size = int(np.round(np.min(np.shape(pic)))/6)
    
    #dilate the image
    dilated = ski.morphology.dilation(pic, ski.morphology.square(grid_size))
    #get the locations of pixels where dilated image equals original
    locs_r,locs_c = np.where(dilated==pic)
    #get values of local_maxima
    vals = pic[locs_r,locs_c]
    
    #plot local maxima
    if plot == True:
        plt.imshow(pic, cmap = 'gray')
        plt.plot(locs_c,locs_r,'o',color = color, markersize = 5, alpha = 0.8);
    return locs_r,locs_c,vals

def thresher(pic, threshold):
    thresh = np.where(pic < threshold, 0, 1) #binarizes the image
    return thresh

def sobel_kernel(shape, axis):
    #finds sobel kernel of given shape
    k = np.zeros(shape)
    p = [(j,i) for j in range(shape[0]) 
           for i in range(shape[1]) 
           if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]

    for j, i in p:
        j_ = int(j - (shape[0] -1)/2.)
        i_ = int(i - (shape[1] -1)/2.)
        k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)
    #flips the kernel for positive values in the first column/row
    k = np.flip(k, (axis+1)%2)
    return k
