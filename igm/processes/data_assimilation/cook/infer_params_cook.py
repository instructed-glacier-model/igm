import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
  
def infer_params_cook(state, cfg):
    #This function allows for both parameters to be specified as varying 2D fields (you could compute them pixel-wise from VelMag by swapping in VelMag for VelPerc).
    #This is probably not a good idea, because the values of both parameters do not depend solely on the velocity at that point. But feel free to try! If you do
    #want to do that, you'll also need to un-comment the code block for smoothing and then converting the smoothed weights back to tensors (you may also want to
    #visualise the figures!), and set up a state.convexity_weights field to act as the 2D array
    import scipy
    
    #Get list of G entities in each C/get multi-valued ice mask
    #Loop over all Gs to construct parameter rasters
    
    percthreshold = 99
    NumGlaciers = int(tf.reduce_max(state.icemask).numpy())
    state.convexity_weights = tf.experimental.numpy.copy(state.icemaskobs)
    state.volumes = tf.experimental.numpy.copy(state.icemaskobs)
    state.volume_weights = tf.experimental.numpy.copy(state.icemaskobs)
    state.volume_weights = tf.where(state.icemaskobs > 0, cfg.processes.data_assimilation.opti_vol_std, 0.0)
    
    
    #Get some initial information
    VelMag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    VelMag = tf.where(tf.math.is_nan(VelMag),1e-6,VelMag)
    VelMag = tf.where(VelMag==0,1e-6,VelMag)
    
    AnnualTemp = tf.reduce_mean(state.air_temp, axis=0)
    
    grady = tf.experimental.numpy.diff(state.usurfobs,axis=0)
    gradx = tf.experimental.numpy.diff(state.usurfobs,axis=1)
    
    #Pad gradients to have same size as DEM input
    paddingsx = tf.constant([[0,0],[1,1]])
    paddingsy = tf.constant([[1,1],[0,0]])
    
    grady = tf.pad(grady,paddingsy)
    gradx = tf.pad(gradx,paddingsx)
    
    #Calculate average gradient value per cell
    Slopes = (tf.math.abs(grady[1:,:])+tf.math.abs(grady[:-1,:])+tf.math.abs(gradx[:,1:])+tf.math.abs(gradx[:,:-1]))/4
    #Smooth slopes
    # Slopes = Slopes.numpy()
    # Slopes = scipy.signal.spline_filter(Slopes, lmbda=1000)
    # Slopes = tf.convert_to_tensor(Slopes)
    
    #Start of G loop
    for i in range(1,NumGlaciers+1):
        
        #Get area predictor
        Area = tf.reduce_sum(tf.where(state.icemask==i,1.0,0.0))*state.dx**2
        Area = np.round(Area.numpy()/1000**2, decimals=1)
        #print('Area is: ', Area)
        #print('Predicted volume is: ', np.exp(np.log(Area)*1.359622487))
        #Work out nominal volume target based on volume-area scaling - only has much of an effect if no other observations
        state.volumes = tf.where(state.icemaskobs == i, np.exp(np.log(Area)*1.354395785), state.volumes) #Make sure to put into km3!
        if Area <= 0.0:
            continue
        
        #Get velocity predictors
        VelMean = np.round(np.mean(VelMag[state.icemaskobs==i]),decimals=2)
        #print("Mean velocity is: ", VelMean)
        VelPerc = np.round(np.percentile(VelMag[state.icemaskobs==i], percthreshold))
        #print("99th percentile velocity is: ", VelPerc)
        
        if VelMean == 0.0:
            #state.icemask = tf.where(state.icemask == i, -1, state.icemask)
            #state.icemaskobs = tf.where(state.icemaskobs == i, -1, state.icemask)
            
            #With volume-area scaling (on the assumption these will all be very small)
            state.convexity_weights = tf.where(state.icemaskobs == i, 5.0, state.convexity_weights)
            state.slidingco = tf.where(state.icemaskobs == i, 0.1, state.slidingco)
            state.volume_weights = tf.where(state.icemaskobs == i, 0.001, state.volume_weights)
            
            continue
        
        #Get average annual air temperature across whole domain
        AvgTemp = tf.reduce_mean(AnnualTemp[state.icemaskobs==i]).numpy()
        AvgTemp = np.round(abs(AvgTemp), decimals=1)
        #print(AvgTemp)
        
        #Get thickness predictors (this uses Millan et al. (2022) thickness estimates just to give some idea of the expected ice thicknesses)
        MaxThk = tf.math.reduce_max(state.thkinit[state.icemask==i])
        MeanThk = tf.math.reduce_mean(state.thkinit[state.icemask==i])
        MaxThk = tf.math.round(MaxThk)
        MeanThk = tf.math.round(MeanThk)
        #print('Max and mean thickness are: ',MaxThk, MeanThk)
        
        #Get slope field
        #Compute spatial gradients of DEM
        #AvgSlope = np.round(tf.reduce_mean(Slopes[state.icemaskobs==i]).numpy(), decimals=1)
        AvgSlope = np.round(tf.reduce_max(state.slopes[state.icemaskobs==i]).numpy(), decimals=1)
        #print("Average Slope is: ", AvgSlope)
        
        Tidewater = cfg.processes.data_assimilation.opti_tidewater_glacier
        
        #Do regressions
        if hasattr(state, "tidewatermask"):
            if tf.reduce_max(state.tidewatermask[state.icemask == i]).numpy() == 1:
                Tidewater = True
            
        if Tidewater == True:
        
            #print('Tidewater')
            #Set up various constants
            VelMean1DCW = VelMean*-32.54498757
            Slope1DCW = AvgSlope*78.90165628
            
            MeanThkCW = MeanThk*2.39253558
            MaxThkCW = MaxThk*2.06356416
            AreaCW = Area*-0.671463838
            MaxVelCW = VelPerc*5.728069133
            MeanVelCW = VelMag*-32.54498757
            SlopeCW = Slopes*78.90165628
            
            VelMean1DSC = tf.math.log(VelMean)*0.779646843
            #Slope1DSC = tf.math.log(AvgSlope)*-0.075281289
            
            LogMeanThkSC = tf.math.log(MeanThk)*2.113449074
            LogTempSC = np.log(AvgTemp)*-0.305778358
            LogMaxThkSC = tf.math.log(MaxThk)*-1.45777084
            LogAreaSC = np.log(Area)*0.220126846
            LogMaxVelSC = np.log(VelPerc)*-1.003939161
            LogMeanVelSC = tf.math.log(VelMag)*0.779646843
            LogMeanVelSC = tf.where(tf.math.is_nan(LogMeanVelSC),0,LogMeanVelSC)
                    
            #These regressions are purely empirical and are based on a selection of 37 glaciers from around the world with thickness measurements,
            #where IGM inversions were performed and the best parameters chosen
            
            CW1D = MeanThkCW + MaxThkCW + AreaCW + MaxVelCW + VelMean1DCW + Slope1DCW - 1323.461811
            SC1D = tf.math.exp(LogMeanThkSC + LogTempSC + LogMaxThkSC + LogAreaSC + LogMaxVelSC + VelMean1DSC - 3.157484542)
            #print(CW1D.numpy(),SC1D.numpy())
            
            #state.convexity_weights[state.icemaskobs==i] = AreaCW + TempCW + MaxVelCW + MeanThkCW + SlopeCW + MeanVelCW + MaxThkCW + 6797.266974
            #state.slidingco[state.icemaskobs==i] = tf.math.exp(LogMeanThkSC + LogTempSC + LogAreaSC + LogMaxVelSC + LogMeanVelSC + LogMaxThkSC + LogSlopeSC + 31.04050337)
            
            #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)
            maxconvexityweight = 10000.0
            minconvexityweight = 5.0
            
            maxslidingco = 0.1
            minslidingco = 0.01
            
            state.convexity_weights = tf.where(state.icemaskobs == i, CW1D.numpy(), state.convexity_weights)
            state.convexity_weights = tf.where(state.convexity_weights > maxconvexityweight, maxconvexityweight, state.convexity_weights)
            state.convexity_weights = tf.where(state.convexity_weights < minconvexityweight, minconvexityweight, state.convexity_weights)
            
            state.slidingco = tf.where(state.icemaskobs == i, SC1D.numpy(), state.slidingco)
            state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
            state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)      
           
            #Smooth both fields a lot
            ConvexityWeights = state.convexity_weights.numpy()
            SlidingCo = state.slidingco.numpy()
            
            #ConvexityWeights = scipy.signal.spline_filter(ConvexityWeights, lmbda=500000)
            #SlidingCo = scipy.signal.spline_filter(SlidingCo, lmbda=500000)
        else:
            if (AvgTemp < 7.0) & (AvgTemp > 0.0):
                #Set up various constants
                VelMean1DCW = tf.math.log(VelMean)*-188.4784857
                Slope1DCW = tf.math.log(AvgSlope)*158.3028365
                
                LogTempCW = np.log(AvgTemp)*-311.5007094
                LogMeanThkCW = tf.math.log(MeanThk)*385.4459893
                LogMeanVelCW = tf.math.log(VelMag)*-188.4784857
                LogMeanVelCW = tf.where(tf.math.is_nan(LogMeanVelCW),0,LogMeanVelCW)
                LogSlopeCW = tf.math.log(Slopes)*158.3028365
                LogSlopeCW = tf.where(tf.math.is_nan(LogSlopeCW),0,LogSlopeCW)
                LogAreaCW = np.log(Area)*248.6125172
                LogMaxThkCW = tf.math.log(MaxThk)*-512.6524209
                Log99VelCW = np.log(VelPerc)*119.2131799
                
                VelMean1DSC = VelMean*-0.000548089
                Slope1DSC = AvgSlope*0.000638754
                
                MeanVelSC = VelMag*-0.000548089
                SlopeSC = Slopes*0.000638754
                AreaSC = Area*-0.000902692
                MaxThkSC = MaxThk*0.000154141
                MaxVelSC = VelPerc*0.0000736281
                        
                #These regressions are purely empirical and are based on a selection of 37 glaciers from around the world with thickness measurements,
                #where IGM inversions were performed and the best parameters chosen
                
                #This fills the area outside the ice mask with the correct average inferred parameter values (so the smoothing works properly)
                CW1D = LogTempCW + LogMeanThkCW + VelMean1DCW + Slope1DCW + LogAreaCW + LogMaxThkCW + Log99VelCW + 798.3035279
                SC1D = VelMean1DSC + Slope1DSC + AreaSC + MaxThkSC + MaxVelSC + 0.021767969
                #print(CW1D.numpy(),SC1D.numpy())
                
                #state.convexity_weights[state.icemaskobs==i] = LogMaxThkSC + LogMeanVelCW + LogMeanThkCW + LogSlopeCW + LogAreaCW + Log99VelCW + LogTempCW + 801.9143279
                #state.slidingco[state.icemaskobs==i] = MaxThkSC + MeanVelSC + SlopeSC + AreaSC + TempSC + MaxVelSC + 0.017414339
                
                #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)
                maxconvexityweight = 1500.0
                minconvexityweight = 5.0
                
                maxslidingco = 0.1
                minslidingco = 0.01
                
                state.convexity_weights = tf.where(state.icemaskobs == i, CW1D.numpy(), state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights > maxconvexityweight, maxconvexityweight, state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights < minconvexityweight, minconvexityweight, state.convexity_weights)
                
                state.slidingco = tf.where(state.icemaskobs == i, SC1D.numpy(), state.slidingco)
                state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
                state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)      
               
                #Smooth both fields a lot
                ConvexityWeights = state.convexity_weights.numpy()
                SlidingCo = state.slidingco.numpy()
                
                #ConvexityWeights = scipy.signal.spline_filter(ConvexityWeights, lmbda=500000)
                #SlidingCo = scipy.signal.spline_filter(SlidingCo, lmbda=500000)
                
            elif (AvgTemp >= 7.0) & (AvgTemp < 15.0):
                #Set up various constants        
                VelMean1DCW = VelMean*6.860440721
                #Slope1DCW = AvgSlope*7.076120236
                
                TempCW = AvgTemp*13.14963315
                MeanVelCW = VelMag*6.860440721
                AreaCW = Area*0.270785203
                MeanThkCW = MeanThk*-2.497513556
                MaxThkCW = MaxThk*1.190856765
                
                VelMean1DSC = VelMean*-0.006108525
                Slope1DSC = AvgSlope*0.004894406
                
                MaxVelSC = VelPerc*0.001061351
                SlopeSC = Slopes*0.004894406
                TempSC = AvgTemp*0.004704947
                MeanVelSC = VelMag*-0.006108525
                AreaSC = Area*-0.00044726
                MeanThkSC = MeanThk*0.001378985
                        
                #These regressions are purely empirical and are based on a selection of 37 glaciers from around the world with thickness measurements,
                #where IGM inversions were performed and the best parameters chosen
                
                CW1D = TempCW + VelMean1DCW + AreaCW + MeanThkCW + MaxThkCW - 102.6704589
                SC1D = MaxVelSC + Slope1DSC + TempSC + VelMean1DSC + AreaSC + MeanThkSC - 0.153944193
                #print(CW1D.numpy(),SC1D.numpy())
                
                #state.convexity_weights[state.icemaskobs==i] = MeanThkCW + MaxThkCW + MeanVelCW + AreaCW + MaxVelCW + TempCW - 32.12685937
                #state.slidingco[state.icemaskobs==i] = MaxVelSC + MeanThkSC + MaxThkSC + MeanVelSC + AreaSC + SlopeSC + TempSC - 0.028584459
                
                #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)
                maxconvexityweight = 1500.0
                minconvexityweight = 5.0
                
                maxslidingco = 0.1
                minslidingco = 0.01
                
                state.convexity_weights = tf.where(state.icemaskobs == i, CW1D.numpy(), state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights > maxconvexityweight, maxconvexityweight, state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights < minconvexityweight, minconvexityweight, state.convexity_weights)
                
                state.slidingco = tf.where(state.icemaskobs == i, SC1D.numpy(), state.slidingco)
                state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
                state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)      
               
                #Smooth both fields a lot
                ConvexityWeights = state.convexity_weights.numpy()
                SlidingCo = state.slidingco.numpy()
                
                # ConvexityWeights = scipy.signal.spline_filter(ConvexityWeights, lmbda=500000)
                # SlidingCo = scipy.signal.spline_filter(SlidingCo, lmbda=500000)
                
            elif AvgTemp >= 15.0:
                #Set up various constants
                VelMean1DCW = tf.math.log(VelMean)*1.853792604
                Slope1DCW = tf.math.log(AvgSlope)*1.486422786
                
                LogMeanThkCW = tf.math.log(MeanThk)*0.41223867
                LogSlopeCW = tf.math.log(Slopes)*1.486422786
                LogSlopeCW = tf.where(tf.math.is_nan(LogSlopeCW),0,LogSlopeCW)
                LogMeanVelCW = tf.math.log(VelMag)*1.853792604
                LogMeanVelCW = tf.where(tf.math.is_nan(LogMeanVelCW),0,LogMeanVelCW)
                LogMaxThkCW = tf.math.log(MaxThk)*0.31532837
                LogMaxVelCW = np.log(VelPerc)*-1.542490077
                LogAreaCW = np.log(Area)*0.307141412
                
                VelMean1DSC = tf.math.log(VelMean)*0.008472703
                Slope1DSC = tf.math.log(AvgSlope)*-0.02595954
                
                LogTempSC = np.log(AvgTemp)*0.097948183
                LogMeanThkSC = tf.math.log(MeanThk)*-0.005875668
                LogSlopeSC = tf.math.log(Slopes)*-0.02595954
                LogSlopeSC = tf.where(tf.math.is_nan(LogSlopeSC),0,LogSlopeSC)
                LogMeanVelSC = tf.math.log(VelMag)*0.008472703
                LogMeanVelSC = tf.where(tf.math.is_nan(LogMeanVelSC),0,LogMeanVelSC)
                LogMaxThkSC = tf.math.log(MaxThk)*-0.010325374
                LogMaxVelSC = np.log(VelPerc)*-0.013272251
                        
                #These regressions are purely empirical and are based on a selection of 37 glaciers from around the world with thickness measurements,
                #where IGM inversions were performed and the best parameters chosen
                
                CW1D = tf.math.exp(LogMeanThkCW + Slope1DCW + VelMean1DCW + LogMaxThkCW + LogMaxVelCW + LogAreaCW - 1.681655194)
                SC1D = LogTempSC + LogMeanThkSC + Slope1DSC + VelMean1DSC + LogMaxThkSC + LogMaxVelSC - 0.072757514
                #print(CW1D.numpy(),SC1D.numpy())
                
                #state.convexity_weights[state.icemaskobs==i] = tf.math.exp(LogMeanVelCW + LogTempCW + LogMeanThkCW + LogMaxThkCW - 5.36050456)
                #state.slidingco[state.icemaskobs==i] = MaxVelSC + SlopeSC + AreaSC + MeanVelSC + MeanThkSC + MaxThkSC + 0.092110566
                
                #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)
                maxconvexityweight = 1500.0
                minconvexityweight = 5.0
                
                maxslidingco = 0.1
                minslidingco = 0.01
                
                state.convexity_weights = tf.where(state.icemaskobs == i, CW1D.numpy(), state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights > maxconvexityweight, maxconvexityweight, state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights < minconvexityweight, minconvexityweight, state.convexity_weights)
                
                state.slidingco = tf.where(state.icemaskobs == i, SC1D.numpy(), state.slidingco)
                state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
                state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)      
               
                #Smooth both fields a lot
                ConvexityWeights = state.convexity_weights.numpy()
                SlidingCo = state.slidingco.numpy()
                
                #ConvexityWeights = scipy.signal.spline_filter(ConvexityWeights, lmbda=500000)
                #SlidingCo = scipy.signal.spline_filter(SlidingCo, lmbda=500000)
        
    #End of G loop
    #To plot weights if required
    # VolWeights = state.volume_weights.numpy()
    # VolumesNumpy = state.volumes.numpy()
    # fig = plt.figure(2, figsize=(8, 7),dpi=200) 
    # plt.subplot(2, 1, 1)
    # plt.imshow(VolWeights, cmap='jet',origin='lower')
    # plt.colorbar(label='volumeweights')
    # plt.title('volumeweights') 
    # plt.xlabel('Distance, km') 
    # plt.ylabel('Distance, km') 

    # plt.subplot(2, 1, 2)
    # plt.imshow(VolumesNumpy, cmap='jet',origin='lower')
    # plt.colorbar(label='volumes')
    # plt.title('volumes') 
    # plt.xlabel('Distance, km') 
    # plt.ylabel('Distance, km') 
    # plt.show()
    
    # ConvexityWeights = tf.convert_to_tensor(ConvexityWeights)
    # SlidingCo = tf.convert_to_tensor(SlidingCo)
    
    # state.convexity_weights = ConvexityWeights
    # state.slidingco = SlidingCo
    
    # #Reapply limiters in case smoothing has messed things up a bit
    # state.convexity_weights = tf.where(state.convexity_weights > maxconvexityweight, maxconvexityweight, state.convexity_weights)
    # state.convexity_weights = tf.where(state.convexity_weights < minconvexityweight, minconvexityweight, state.convexity_weights)
    
    # state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
    # state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)