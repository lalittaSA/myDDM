%% new driff diffusion model adapted from Yunshu

function [fits,err] = Script_fit_RT_DDM1(options)
%20170214 created

if nargin < 1
    options.optionsName = 'default';          
end
    


%%%%%%%%%%%%%
% set paths %
%%%%%%%%%%%%%

data_folder = '/Research/uPenn_auditoryDecision/data/psychophysics/';
result_folder = '/Research/uPenn_auditoryDecision/results/psychophysics/';

ST = dbstack;
funcName = ST.name;
result_folder = checkDirectory(result_folder,['results_' funcName],1);

listing_dataFile = dir([data_folder,'*_table.mat']);
nFiles = length(listing_dataFile);
filenames = cell(nFiles,1);

% %%%%%%%%%%%%%%%%%%%%%%%%%
% % set default variables %
% %%%%%%%%%%%%%%%%%%%%%%%%%
% 
% defaultOptions.optionsName = 'default';
% defaultOptions.display.visualize = true;                                           % show visual representation
% defaultOptions.display.recordMovie = false;                                        % store visualization as avi-movie
% defaultOptions.display.movieFile = [funcName '_movie.avi'];                        % file name for movie
% 
% defaultOptions.taskVar.coherenceList = [-0.512 -0.256 -0.128 -0.064 -0.032 0 0.032 0.064 0.128 0.256 0.512];
% defaultOptions.taskVar.n_rep = 1000;                                     % number of trials for each coherence level
% defaultOptions.taskVar.trialLength = 1000;                                 % number of accumulation steps
% 
% defaultOptions.modelVar.threshold_a = 20;
% defaultOptions.modelVar.threshold_b = 20;
% defaultOptions.modelVar.k = 0.03;
% defaultOptions.modelVar.t0_a = 200;
% defaultOptions.modelVar.t0_b = 200;
% defaultOptions.modelVar.dME = 0.05;

% go through all data files
for ii = 1:nFiles
    filenames{ii} = listing_dataFile(ii).name;
    savename = [funcName '_' filenames{ii}];
    load([data_folder filenames{ii}]);
    
     % rearrange a bit
    data_table.success(isnan(data_table.success)) = 0;
    data_table.success = logical(data_table.success);
    ind_nan = isnan(data_table.choice) | isnan(data_table.RT);
    prior_list = unique(data_table.priorLevel);
    data_table.prior_id = data_table.priorLevel;
    prior_name = cell(length(prior_list),1);
    nPrior = length(prior_list);
    for pp = 1:nPrior
        ind = data_table.priorLevel == prior_list(pp);
        data_table.prior_id(ind) = pp;
        switch prior_list(pp)
            case -3,    prior_name{pp} = 'Lo 5:0';
            case -2,    prior_name{pp} = 'Lo 4:1';
            case -1,    prior_name{pp} = 'Lo 3:2';
            case 0,     prior_name{pp} = '1:1';
            case 1,     prior_name{pp} = '2:3 Hi';
            case 2,     prior_name{pp} = '1:4 Hi';
            case 3,     prior_name{pp} = '0:5 Hi';
        end
    end
    data_table.prior_text = prior_name(data_table.prior_id);
    choice_name = {'none';'right (Lo)';'left (Hi)'};
    tmp_choice = data_table.choice+1;
    tmp_choice(isnan(tmp_choice)) = 1;
    data_table.choice_text = choice_name(tmp_choice);
    success_name = {'correct';'incorrect'};
    
    data_table.coh_played_norm = (data_table.coh_played - 0.5)*2;
    [Y,E] = discretize(data_table.coh_played_norm,-1:0.2:1);
    mid_bins = E(1:end-1)'+(E(2)-E(1))/2;
    data_table.coh_group = mid_bins(Y);
    
    x0 = rand(10,1);
    [fits,err, ~, output] = fminsearch(@(x)DDM_spectral_err1(x,data_table.coh_group,data_table.choice,data_table.RT,data_table.priorLevel), x0, ...
        optimset('Display', 'Off','PlotFcn',@optimplotfval,'TolFun',1e-3));
    
end
end

function err = DDM_spectral_err1(x, cohs, choices, rts, prior)
% x = [threshold_up, threshold_lo, scalingfactor, ndT, ndT, ME bias, ndT, ndT, ME bias, starting value];

cohlist = unique(cohs);
dt = 1;  % 1 ms. Unit ms

k_rew0 = x(3);
Blo_rew0 = -x(2)+x(10);
Bup_rew0 = x(1)+x(10);
C_rew0 = cohlist+x(6);

T01_rew0 = x(4);
T02_rew0 = x(5);
T01_rew1 = x(7);
T02_rew1 = x(8);

tmax_prior0 = ceil(max(rts(prior==0)));
t_prior0 = (0:dt:tmax_prior0)'; % decision time from 0 to tmax ms

D_rew0 = get_pdf_spectral(C_rew0,k_rew0,Blo_rew0,Bup_rew0,dt,t_prior0);
pdf_t_up_rew0 = D_rew0.up.pdf_t;
pdf_t_lo_rew0 = D_rew0.lo.pdf_t;

err_rew0 = llr_spectral(pdf_t_up_rew0, pdf_t_lo_rew0, cohs(prior==0), cohlist, choices(prior==0), rts(prior==0), T01_rew0, T02_rew0);

k_rew1 = x(3);
Blo_rew1 = -x(2)-x(10);
Bup_rew1 = x(1)-x(10);
C_rew1 = cohlist + x(9);

tmax_rew1 = ceil(max(rts(prior==3)));
t_rew1 = (0:dt:tmax_rew1)'; % decision time from 0 to tmax ms

D_rew1 = get_pdf_spectral(C_rew1,k_rew1,Blo_rew1,Bup_rew1,dt,t_rew1);
pdf_t_up_rew1 = D_rew1.up.pdf_t;
pdf_t_lo_rew1 = D_rew1.lo.pdf_t;

err_rew1 = llr_spectral(pdf_t_up_rew1, pdf_t_lo_rew1, cohs(prior==3), cohlist, choices(prior==3), rts(prior==3), T01_rew1, T02_rew1);

err = err_rew0 + err_rew1;

end

function D = get_pdf_spectral(C,scalingfactor,Blo,Bup,dt,t)
% Yunshu 2014-1-19. force C0 to be a column vector
    
    drift = C * scalingfactor; 
    % sm: a small overhanging of y
    sm = max(abs(drift))*dt+4*sqrt(dt);
    y = linspace(min(Blo)-sm,max(Bup)+sm,1024)'; % could be 1024*2
    % construct the PDF for initial value
    % if y contains SVbias, then delta function at SVbias
    % if y does not contain SVbias, then split for the two values closest to SVbias based on their distance to SVbias;
    y0 = 0*y;
    i1=find(y>=0, 1,'first');
    i2=find(y<=0, 1,'last');
    if i1==i2
        y0(i1)=1;
    else
        w = abs(y([i1 i2]));
        w = w/(sum(w));
        y0(i1) = w(1);
        y0(i2) = w(2);
    end
    [D,~] =  spectral_dtbAAsigma_LD(drift,t,Bup,Blo,y,y0, 0, 1, 0);

end

function err = llr_spectral(pdf_t_up,pdf_t_lo,cohs, cohlist, choices, rts, T01, T02)
ind_up = (choices == 1);
ind_lo = (choices == 2);
% rt_round = round(rts);
Tnd = (ind_up*T01 + ind_lo*T02)*1000;
decisionT = round(rts-Tnd);
if sum(decisionT<0)>1
    err = 10000;
else
    ncohs = length(cohlist);
    coh_ind_temp = nan(size(cohs,1),ncohs);
    for icoh = 1:ncohs
        coh_ind_temp(:,icoh) = (cohs == cohlist(icoh))*icoh;
    end
    coh_ind = sum(coh_ind_temp,2);

    ind_lin_up = sub2ind(size(pdf_t_up),decisionT(ind_up),coh_ind(ind_up));
    % rt_up = rt_round(ind_up);coh_up = coh_ind(ind_up);
    % ind_lin_up = sub2ind(size(pdf_t_up),rt_up(1:4),coh_up(1:4));
    ind_lin_lo = sub2ind(size(pdf_t_lo),decisionT(ind_lo),coh_ind(ind_lo));
    LH_up_each = pdf_t_up(ind_lin_up);
    LH_lo_each = pdf_t_lo(ind_lin_lo);
    if (sum(LH_up_each) == 0) || (sum(LH_lo_each)==0)
        err = 10000;
    else
        LLH = sum(log(LH_up_each(LH_up_each~=0)))+sum(log(LH_lo_each(LH_lo_each~=0)));
        err = -LLH;
    end
end

end
