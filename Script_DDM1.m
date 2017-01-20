%% new driff diffusion model adapted from Yunshu

function Script_DDM1(options)

if nargin < 1
    options.optionsName = 'default';          
end
    
%%%%%%%%%%%%%%%%%%%
% set result path %
%%%%%%%%%%%%%%%%%%%

ST = dbstack;
funcName = ST.name;
resultPath = checkDirectory(['.' filesep],['results_' funcName],1);

%%%%%%%%%%%%%%%%%%%%%%%%%
% set default variables %
%%%%%%%%%%%%%%%%%%%%%%%%%

defaultOptions.optionsName = 'default';
defaultOptions.display.visualize = true;                                           % show visual representation
defaultOptions.display.recordMovie = false;                                        % store visualization as avi-movie
defaultOptions.display.movieFile = [funcName '_movie.avi'];                        % file name for movie

defaultOptions.taskVar.coherenceList = [-0.512 -0.256 -0.128 -0.064 -0.032 0 0.032 0.064 0.128 0.256 0.512];
defaultOptions.taskVar.n_rep = 1000;                                     % number of trials for each coherence level
defaultOptions.taskVar.trialLength = 1000;                                 % number of accumulation steps

defaultOptions.modelVar.threshold_a = 20;
defaultOptions.modelVar.threshold_b = 20;
defaultOptions.modelVar.k = 0.03;
defaultOptions.modelVar.t0_a = 200;
defaultOptions.modelVar.t0_b = 200;
defaultOptions.modelVar.dME = 0.05;

options = setScriptOptions(defaultOptions, options); 
clear defaultOptions

cohList = options.taskVar.coherenceList;
n_coh = length(cohList);
n_rep = options.taskVar.n_rep;
n_step = options.taskVar.trialLength;

n_trial = n_rep * n_coh;
noise = randn(n_trial,n_step);

coh = ((cohList(:)) * ones(1,n_rep))';
coh = coh(:);

d_a = coh > 0;
d_b = coh < 0;
dir = d_a + (d_b * 2);

k = options.modelVar.k;
momentEvidence = coh * k * ones(1,n_step) + noise;
dv = cumsum(momentEvidence,2);

threshold_a = options.modelVar.threshold_a;
threshold_b = options.modelVar.threshold_b;
t0_a = options.modelVar.t0_a;
t0_b = options.modelVar.t0_b;

rt_a = sum(((cumsum((dv > threshold_a),2)) == 0),2);
rt_b = sum(((cumsum((dv < -threshold_b),2)) == 0),2);

choice_a = rt_a < rt_b;
choice_b = rt_b < rt_a;
undecide = rt_b == rt_a; % check number of undecided trials
rt = (rt_a + t0_a).*choice_a + (rt_b + t0_b).*choice_b;
rt(rt == 0) = nan;

choice = choice_a + (choice_b * 2);
correct = (dir == choice);

trialTable = table(coh,dir,choice,correct,rt,'VariableNames',{'coherenceLevel','direction','choice','correct','rt'});

avg_choice = grpstats(trialTable,{'coherenceLevel','correct'},'mean','DataVars',{'choice'});
avg_rt = grpstats(trialTable,{'coherenceLevel','correct','choice'},'nanmean','DataVars',{'rt'});

