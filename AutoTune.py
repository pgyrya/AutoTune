####################################################################################################
# implementation of step-by-step parameter search
# steps are generated randomly
#
# Changes as of Feb 1, 2018
# - ensured pseudo-randomization of each step by relying on numpy package functionality,
# seeded with fraction of system time (share of milliseconds)
# - allowed string valued hyperparameters, updates governed by dictionary param_options_str
# - allowed steps along multiple dimensions
# - saved step details information in results dictionary (what changed, score improvement)
# - introduced an option to limit maximum fitting time
# Additional future changes considered (may not be listed on github)
# - extend dictionary of prior results with new values, if merging with prior 
#
# Changes as of Feb 22 - Mar 1, 2018
# - passed on extra fit parameters (e.g. number of epochs, batch size) - to gridsearch fit 
# - saved time improvement parameter
# - created save_best_result function
# - setup trade-off between incremental performance improvement and calibration time 
# - setup learners to anticipate performance and time commitment based on varibales that changed
#   note this could be modified in the future to anticipate changes instead of values themselves
# - adjusted random step selection to prioritize steps anticipated to improve performance trade-off
#   balancing exploration and exploitation
# - augmented results file with additional fields (prior parameters, incremental performance, time)
####################################################################################################
# In[ ]
import copy, math
import numpy as np
import pandas as pd
import gc
import copy
import time

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# decided not to inherit it for now from GridSearchCV for now,
# instead creating an instance of this class internally
class StepSearchCV():    
    def __init__(
        self,
        estimator        = None,                       # reference to a base estimator, or pipeline of estimators
        parameters_start = None,                       # initial parameters
        param_options_str= None,                       # dictionary providing a list of options for each string hyperparameter
        prior_results    = None,                       # optional dictionary with prior results, to be extended with new results of the same type, or object of the same class
        prior_best_score = None,                       # optional best model score, as calibrated on prior results        
        step_number_vars = 1,                          # number of dimensions to adjust per step
        step_rules       = None,                       # optional dictionary containing two scales - for random additive and multiplicative transformations, see default below
        step_veto_rule   = None,                       # optional function of current and new value that returns True if the step is not interesting, see default below
        steps            = math.inf,                   # number of randopm steps requested during initialization, could be over-written during fit() method call
        step_threshold   = 0.0001,                     # set to positive if higher model performance score is considered better; should indicate sign of performance improvement
        scoring          = None,                       # if None the search would rely on estimator model class internal method score(self, X, y)
        random_state     = 7,                          # not currently implemented - intended as a seed to be supplied to the estimator fit function  
        save_location    = '../output/cv_results.csv', # by default exporting to the parent folder, as assuming current directory would be 'code'
        cv_folds         = 2,                          # number of folds (K) to be used for K-fold cross-validation 
        refit            = False,                      # whether to re-fit each model on the combined set 
        reset_session    = True,                       # whether to re-set each Keras session after every iteration
        extending_lists  = True,                       # whethyr list and tuple parameters could be extended
        delete_prior     = False,                      # object of the same class to inherit search results from
        max_fit_time     = None,                       # maximum fitting time allowed, models would be rejected if it is exceeded on average
        meta_learner     = None,                       # estimator to be used for anticipating changes in performance and resource cost in response to random steps
        explore_rt_init  = 1.0,                        # by default, focus 100 percent on exploration initially, with below decay rate
        explore_rt_decay = 0.25,                        # inverse linear decay rate of .25 means it takes 4 steps to decay rate by factor of 2, 8 steps by 3, etc.
        explore_rt_min   = 0.5,                        # minimum exploration rate - especially helpful if anticipation models are often wrong
        anticipation_draws   = 25,                     # number of random draws to be evaluated based on anticipation models 
        anticipate_changes   = False,                  # for future use - it may be helpful to anticipate changes instead of values
        tradeoff_scr_per_sec = 0.,                     # 1 unit of score improvement is considered exchangeable for tradeoff_scr_per_sec seconds of fitting time
        ):
        
        # initialization
             
        # adjust default inputs, if not provided
        if step_rules     == None: step_rules = {'multiply_scale': 1.0, 'add_scale': 0}
        if step_veto_rule == None: step_veto_rule = lambda x,y: y<=0 or(x >=0.95*y) and (x <=1.05*y)
        
            
        # save settings provided with constructor internally
        self.estimator           = estimator
        self.results             = None        
        self.best_score          = prior_best_score        
        self.parameters_best     = parameters_start
        
        self.step_rules          = step_rules
        self.steps_requested     = steps
        self.step                = 0
        self.step_veto_rule      = step_veto_rule
        self.step_threshold      = step_threshold
        self.save_location       = save_location
        self.scoring             = scoring
        self.cv_folds            = cv_folds
        self.refit               = refit
        self.reset_session       = reset_session
        self.extending_lists     = extending_lists
        self.step_number_vars    = step_number_vars
        self.param_options_str   = param_options_str
        self.max_fit_time        = max_fit_time
        self.explore_rt_init     = explore_rt_init
        self.explore_rt_decay    = explore_rt_decay
        self.explore_rt_min      = explore_rt_min
        self.anticipation_draws  = anticipation_draws 
        self.anticipate_changes  = anticipate_changes
        self.tradeoff_scr_per_sec= tradeoff_scr_per_sec
        self.meta_learner        = meta_learner or sklearn.pipeline.make_pipeline(                
                #sklearn.preprocessing.StandardScaler(),
                #sklearn.preprocessing.MinMaxScaler(),    # could consider to adjust parameter feature_range=(0, 1)
                sklearn.linear_model.Ridge()
                #sklearn.svm.SVR(kernel = 'poly', degree = 2)
                #sklearn.neighbors.KNeighborsRegressor(n_neighbors=25, weights='distance', algorithm='auto', n_jobs=1)
                )
        self.meta_learner_score= copy.deepcopy(self.meta_learner)
        self.meta_learner_time = copy.deepcopy(self.meta_learner)
        
        #initialize grid search object intended to save search results
        self.searcher                      = None

        if prior_results is not None:
            print('Analyzing prior results provided..')
            if hasattr(prior_results,'parameters_best'):
                print('Copying prior results from StepSearchCV object provided')
                # assuming the object provided is of class StepSearchCV and 
                # inherit prior results from the search object provided
                self.best_score               = prior_results.best_score
                self.best_model_mean_fit_time = prior_results.best_model_mean_fit_time
                self.results                  = copy.deepcopy(prior_results.results)
                self.step                     = prior_results.step
                if self.parameters_best is None: self.parameters_best = copy.deepcopy(prior_results.parameters_best)
            else:
                print('Copying prior calibration results from dictionary provided')
                self.results       = copy.deepcopy(prior_results)
                
            if delete_prior == True:
                del prior_results
                gc.collect()
        
        return 

    def time_based_random_positive_integer(self, max_value = 10**6):
        return int(max_value * 1000 * (time.time()% 0.001))

    def adjust_single_parameter(self, parameter_to_adjust, current_value):
        new_value = current_value        
        if type(current_value) is bool:            
            if self.step_number_vars == 1:     # option assuming we want to definitely change each parameter
                new_value = not(current_value)
            else:                              # option if we are OK with not forcing adjustment of each parameter
                new_value = np.random.choice([True,False]) 
        if type(current_value) in (str,np.str_) and self.param_options_str is not None and type(self.param_options_str) is dict and parameter_to_adjust in self.param_options_str.keys():
            # pick a random parameter using provided dictionary with options for string hyperparameters
            current_parameter_options = sorted(list(self.param_options_str[parameter_to_adjust])).copy()
            if current_value in current_parameter_options and self.step_number_vars == 1:
                # remove option to keep parameter constant if we only adjust one parameter at a time
                current_parameter_options.remove(current_value)
            new_value = np.random.choice(current_parameter_options)
            #print(new_value) - returns new value that then mysteriously disappears before being printed
        else:    
            while (self.step_veto_rule(current_value, new_value)):
                step_direction                       = np.random.choice([1,-1])
                step_multiply                        = math.exp(step_direction * abs(np.random.normal(scale = self.step_rules['multiply_scale']))) 
                step_add                             =          step_direction * abs(np.random.normal(scale = self.step_rules['add_scale']     )) 
                new_value                            = type(current_value)(step_multiply * current_value + step_add)
                #print("Attempting to adjust {} from {} to {}".format(parameter_to_adjust,current_value,new_value))
        return new_value
        
    def pick_parameter_step(self):
        # note these steps could become more intelligent later as the system learns what works best
        # This process was made a bit more intelligent based on review of what paremeters were already tried

        # initialize random seed based on system time - providing randomization for step direction
        np.random.seed(self.time_based_random_positive_integer())
        
        # randomly generate next step to explore
        current_parameters                 = self.parameters_best
        parameters_best_anticipated        = self.parameters_best
        score_improvement_anticipated_best = 0
        time_improvement_anticipated_best  = 0
        #selection_approved                = False

        # decide whether to spend resources to explore and gather more hard information or prefer to play safe and exploit current knowledge
        choose_to_explore  = np.random.uniform() < max(self.explore_rt_min, self.explore_rt_init / (1.0 + self.step * self.explore_rt_decay))
        #print(f'Choose to explore: {choose_to_explore}')
        number_of_draws    = max(1,self.anticipation_draws * int(not choose_to_explore))        
        #print(f'Number of draws planned: {number_of_draws    }')
        best_draw_index               = None
        best_anticipated_adjustment   = ""
        
        for draw_index in range(number_of_draws):
            parameters_under_review = copy.deepcopy(current_parameters)
            parameters_to_adjust    = np.random.choice(sorted( [key for key in parameters_under_review.keys()]), size = self.step_number_vars, replace = False)        
    
            # pick a new value for each parameter selected for adjustment
            last_step_parameters_adjusted = "Explore: " if choose_to_explore else ""
            for i,parameter_to_adjust in enumerate(parameters_to_adjust):
                #print(parameter_to_adjust)
                current_parameter_value = parameters_under_review[parameter_to_adjust]            
           
                if type(current_parameter_value) in (float, int, bool, str, np.str_):  
                    parameters_under_review[parameter_to_adjust] = self.adjust_single_parameter(parameter_to_adjust, current_parameter_value)
                    
                elif type(current_parameter_value) in (list, tuple):
                    #cast tuples to list in order to support elementwise assignment
                    sequence_type      = type(current_parameter_value)
                    new_sequence_value = list(current_parameter_value)            
                    
                    # Parameter is a sequence of elements, so we would pick and adjust one of the list elements randomly
                    # Alternatively, we could also augment list with additional entry
                    current_parameter_sequence_length = len(current_parameter_value)
                    parameter_index_to_adjust         = np.random.choice(np.arange(current_parameter_sequence_length + int(self.extending_lists))) #add 1 for extending lists
                    
                    if parameter_index_to_adjust < current_parameter_sequence_length:
                        # adjust selected paramter, 
                        
                        current_sequence_element_value = current_parameter_value[parameter_index_to_adjust]
                        if type(current_sequence_element_value) in (float, int, bool):  
                            # ready to adjust selected element of the sequence
                            new_sequence_element_value                    = self.adjust_single_parameter(parameter_to_adjust, current_sequence_element_value)                    
                            new_sequence_value[parameter_index_to_adjust] = new_sequence_element_value
                    else:
                        #augment the list with one more entry, equal to minimum entry from the prior list
                        new_sequence_element_value    = min(current_parameter_value)
                        new_sequence_value.append(new_sequence_element_value)
    
                    
                    parameters_under_review[parameter_to_adjust] = sequence_type(new_sequence_value) #cast back to tuple if necessary
    
                # augment information about what parameters were adjusted and their new values   
                last_step_parameters_adjusted += str(parameter_to_adjust) + \
                                                      " from " + str(self.format_parameter_values(current_parameter_value)) + \
                                                      " to "   + str(self.format_parameter_values(parameters_under_review[parameter_to_adjust]))
                if i< len(parameters_to_adjust)-1:
                    last_step_parameters_adjusted += ";"
            
                # review anticipated improvement in model performance and change in time to fit the model                
                if self.step >= 2: 
                    model_expected_to_improve_materially, score_improvement, time_improvement = self.anticipate_improvement(
                            current_parameters, parameters_under_review)
                else:              
                    model_expected_to_improve_materially, score_improvement, time_improvement = True,0,0
                
            # tentatively approve proceeding with selected adjustment only if we expect improvement over and above previous
            # note this would be repeated multiple times, as per draw
            if choose_to_explore or \
            (score_improvement_anticipated_best == 0 and draw_index == number_of_draws - 1) or \
            ((score_improvement - score_improvement_anticipated_best) >= (time_improvement - time_improvement_anticipated_best ) * self.tradeoff_scr_per_sec and \
             self.best_model_mean_fit_time - time_improvement < self.max_fit_time):
                score_improvement_anticipated_best = score_improvement
                time_improvement_anticipated_best  = time_improvement
                parameters_best_anticipated        = parameters_under_review
                best_anticipated_adjustment        = last_step_parameters_adjusted
                best_draw_index                    = draw_index

        #print('Best draw occured at index', best_draw_index)
        # Might want to consider also returning parameter changed, index and direction
        return parameters_best_anticipated, best_anticipated_adjustment, score_improvement_anticipated_best  
    
    def fit(self, X, y=None, groups=None, export_improvements_only = False, steps = None, test_starting_parameters = True, **fit_params):
        # TODO overwrite steps completed variable
        if steps is not None and type(steps) is int: self.steps_requested = steps + self.step
        
        # iterate until number of steps left drops to zero, or keyboard interrupt        
        parameters_under_review       = self.parameters_best
        last_step_parameters_adjusted = "Picked original parameters"
        score_improvement_anticipated = 0
        model_improved_materially     = True
        
        while self.step < self.steps_requested:
            #print ("Evaluating: ",  parameters_under_review)
            if test_starting_parameters == False:
                parameters_under_review, last_step_parameters_adjusted, score_improvement_anticipated = self.pick_parameter_step()

            print("Step {} of {}, attempting adjustment {}; Anticipating score improvement of {}".format(
                    self.step+1,
                    self.steps_requested,
                    last_step_parameters_adjusted,
                    self.lround(score_improvement_anticipated)
                    ))
        
            # evaluate selected models and save calibration results internally
            # evaluate model on the parameters selected to be next
            search_grid_wrapper = {key:[item] for key, item in parameters_under_review.items()}
            
            self.searcher = GridSearchCV(
                self.estimator,
                search_grid_wrapper,
                scoring            = self.scoring,
                return_train_score = False,
                cv                 = self.cv_folds,
                refit              = self.refit)
            
            try:
                self.searcher.fit(X, y, groups, **fit_params)
                self.save_best_result(
                        parameters_under_review, 
                        self.searcher.cv_results_, 
                        last_step_parameters_adjusted, 
                        score_improvement_anticipated)   # Save test results internally

                # Export combined estimation results to a file, if warranted
                if (export_improvements_only == False) or (model_improved_materially == True):
                    self.results_df = pd.DataFrame.from_dict(self.results)
                    self.results_df.to_csv(self.save_location, index = False)
            
            #except ResourceExhaustedError as e:
            #    print('Exhausted memory resources while trying to fit the new model in memory... \n', e)
            except MemoryError as e:
                print('Out of memory while trying to calibrate model. \n', e)
            #except ResourceExhaustError as e:
            #    print('Out of GPU resources while trying to calibrate model. \n', e)

            # clean up memory resources after each optimization step
            if self.reset_session == True:
                from keras import backend as K
                K.clear_session()            
                self.searcher = None
                gc.collect()

            # calibrate a model to anticipate future response in model performance and fit time - to changes in parameters
            if self.step > 0: self.build_anticipation_model()
            
            # move on to the next step, picking a random step within hyper-parameter space
            self.step += 1
            parameters_under_review, last_step_parameters_adjusted, score_improvement_anticipated = self.pick_parameter_step()
            
        return self.parameters_best #self.estimator, self.best_estimator
    
    def format_parameter_values(self, value):        
        if type(value) in (str, np.str_, bool, int):
            return value
        elif type(value) is float:
            return self.lround(value)
        elif type(value) in (list,tuple):
            return type(value)(self.format_parameter_values(element) for element in value)

    def lround(self,x,leadingDigits=4): 
        """Return x either as 'print' would show it (the default) 
        or rounded to the specified digit as counted from the leftmost 
        non-zero digit of the number, e.g. lround(0.00326,2) --> 0.0033
        """ 
        assert leadingDigits>=1 
        if leadingDigits==1: 
                return float(str(x)) #just give it back like 'print' would give it
        return float('%.*e' % (int(leadingDigits)-1,x)) #give it back as rounded by the %e format 

    def save_best_result(self, parameters_under_review, results, last_step_parameters_adjusted, score_improvement_anticipated):
        assert len(results['mean_fit_time']) == 1 #assume we tested only one model
        new_score    = results['mean_test_score'][-1]
        new_fit_time = results['mean_fit_time'][-1] 
        
        if self.results is None: 
            #initiate cross-validation results dictionary values
            self.results                                      = results               
            self.results['iteration']                         = np.array(1   , dtype = 'int')
            self.results['improved' ]                         = np.array(True, dtype = 'bool')
            self.results['score_improved_by' ]                = np.array(0.,   dtype = 'float')
            self.results['score_improvement_anticipated' ]    = np.array(0.,   dtype = 'float')
            self.results['time_improved_by' ]                 = np.array(0.,   dtype = 'float')
            self.results['changes'  ]                         = np.array([last_step_parameters_adjusted], dtype = 'str' )                    
            self.best_score                                   = new_score    
            self.best_model_mean_fit_time                     = new_fit_time 
            self.results['params_previous']                   = self.results['params']
            
        else:
            # augment cross-validation results dictionary with new test                
            self.results['iteration']    = np.append(self.results['iteration'],self.step + 1)
            self.results['changes'  ]    = np.append(self.results['changes'  ],[last_step_parameters_adjusted])
            
            for key, item in results.items():
                self.results[key]        = np.append(self.results[key],item) #[0]

            # decide whether the newly calibrated model is worthy 
            model_improved_materially, score_improvement, time_improvement = self.evaluate_improvement(new_score, new_fit_time)
            
            self.results['improved']                       = np.append(self.results['improved']                     , model_improved_materially)
            self.results['score_improved_by']              = np.append(self.results['score_improved_by']            , self.lround(score_improvement))
            self.results['score_improvement_anticipated']  = np.append(self.results['score_improvement_anticipated'], self.lround(score_improvement_anticipated))
            self.results['time_improved_by' ]              = np.append(self.results['time_improved_by' ]            , self.lround(time_improvement ))
            self.results['params_previous']                = np.append(self.results['params_previous'  ]            , self.parameters_best)
            
            if model_improved_materially: # the new model is considered worthy                
                self.best_score               = new_score
                self.best_model_mean_fit_time = new_fit_time
                self.parameters_best          = parameters_under_review
                print ("Imroved score by %.3f adopting parameters: \n"% score_improvement , parameters_under_review)


    def vectorize_parameters(self, parameters_dict = None):
        # identify parameters that had a chance to vary
        #parameters_that_had_a_chance_to_vary = [key for key, value in dictionary.items() for dictionary in parameters_previously_considered.items() if len(dictionary[key])]
        parameters_that_had_a_chance_to_vary = [result_field[6:] for result_field, history in self.results.items() if result_field[0:6]=='param_' and len({str(element) for element in history}) > 1]
        #print('parameters_that_had_a_chance_to_vary: ', parameters_that_had_a_chance_to_vary)
        
        relevant_history_dict = {result_field[6:]: history for result_field, history in self.results.items() if result_field[6:] in parameters_that_had_a_chance_to_vary} 
        
        if parameters_dict == None: 
            # vectorize all parameters previously considered into a single data frame
            parameters_dict = relevant_history_dict
            #pd.DataFrame.from_dict(self.results['params']) #input is a list of dictionaries
            number_of_vectors_expected = self.step + 1
        else:
            number_of_vectors_expected = 1        

        # initialize data frame with vectorized parameters
        parameters_vectorized = pd.DataFrame(columns = parameters_that_had_a_chance_to_vary)

        
        for parameter in parameters_that_had_a_chance_to_vary:            
            tested_values    = relevant_history_dict[parameter].tolist()
            parameter_values = [parameters_dict[parameter]] if number_of_vectors_expected == 1 else tested_values
            parameter_type   = type(parameter_values[-1])
            # numeric parameters  - map to float, this includes boolean parameters 
            if parameter_type is bool:                  
                parameters_vectorized[parameter] = pd.Series(parameter_values)
            if parameter_type in (float, int):                  
                #parameters_vectorized.loc[0,[parameter]] = float(parameters_dict[parameter])
                #parameters_vectorized[parameter] = pd.Series(parameter_values).astype(parameter_type)
                del parameters_vectorized[parameter]
                distinct_values_observed = set(relevant_history_dict[parameter])
                if len(distinct_values_observed) >1:
                    parameters_vectorized[parameter+'_log'   ] = pd.Series(parameter_values).apply(math.log).fillna(math.log(0.001))
                if len(distinct_values_observed) >2:
                    #include quadratic terms to capture non-linearity allowing to recognize local minimum
                    parameters_vectorized[parameter+'_log_sq'] = parameters_vectorized[parameter+'_log']**2
                
            # sequence parameters - with based on max length, assume list elements are numeric
            if parameter_type in (list, tuple):
                #max_sequence_length = max({len(p) for p in parameter_values})
                max_sequence_length = max({len(p) for p in self.results['param_'+parameter]}) #only consider sequence length up to what was included in the meta-model
                
                del parameters_vectorized[parameter]                
                for i in range(max_sequence_length):
                    select_i_th              = lambda l: float(l[i]) if len(l)>= i+1 else None
                    series                   = pd.Series(parameter_values).map(select_i_th)
                    parameter_history        = pd.Series(   tested_values).map(select_i_th)
                    distinct_values_observed = set(parameter_history)
                    if len(distinct_values_observed) >1:
                        parameters_vectorized[parameter+'_' + str(i)+'_na'    ] = series.isna().astype(float)
                        parameters_vectorized[parameter+'_' + str(i)+'_log'   ] = series.fillna(0.001).apply(math.log)
                    if len(distinct_values_observed) >2:                    
                        parameters_vectorized[parameter+'_' + str(i)+'_log_sq'] =(series.fillna(0.001).apply(math.log))**2
            
            # string parameters   - apply one-hot encoding based on values already observed
            # TODO - test below functionality for string inputs
            if parameter_type in (str, np.str_):
                del parameters_vectorized[parameter]                
                distinct_parameter_values_detected = sorted(list(set(self.results['param_'+parameter])))
                #str_vectorizer = CountVectorizer(token_pattern = '.+', vocabulary = )
                
                for parameter_string_value,i in enumerate(distinct_parameter_values_detected):
                    parameters_vectorized[parameter+'_'+ str(i)] = pd.Series(parameter_values).map(lambda label: float(label == parameter_string_value))
                    
        
        return parameters_vectorized
        
    def build_anticipation_model(self, target = 'absolute'):
        # builds a meta-model tasked with anticipating results of calibration
        #   inputs  - list of dictionaries saved in self.results['params']
        #   outputs - mean_test_score, mean_fit_time
        X = self.vectorize_parameters()
        
        if target == 'absolute':
            y_score = self.results['mean_test_score'].reshape((-1))
            y_time  = self.results['mean_fit_time'  ].reshape((-1))
        elif target == 'relative':
            y_score = self.results['score_improved_by'].reshape((-1))
            y_time  = self.results['time_improved_by'  ].reshape((-1))
        
        if hasattr(self.meta_learner.named_steps,'kneighborsregressor'):
            # TODO adjust minkowski metric based on maximum observed sensitivity of performance to a given hyper-parameter
            # manually reduce number of neighbors for initial steps
            if self.meta_learner.named_steps['kneighborsregressor'].n_neighbors > self.step:
                print('setting number of steps to lower')
                #self.meta_learner_score.named_steps['kneighborsregressor'].n_neighbors = self.step
                #self.meta_learner_time.named_steps['kneighborsregressor'].n_neighbors = self.step            
                self.meta_learner_score.set_params(kneighborsregressor__n_neighbors = self.step)
                self.meta_learner_time.set_params (kneighborsregressor__n_neighbors = self.step)
            
        self.meta_learner_score.fit(X,y_score)          
        self.meta_learner_time.fit(X,y_time)  
        
        return
    
    def anticipate_improvement(self, parameters_dict_current, parameters_dict_new):
#        try:
        predictions_current = {
                'time':  self.meta_learner_time.predict (self.vectorize_parameters(parameters_dict_current)),
                'score': self.meta_learner_score.predict(self.vectorize_parameters(parameters_dict_current))}
        predictions_new     = {
                'time':  self.meta_learner_time.predict (self.vectorize_parameters(parameters_dict_new)),
                'score': self.meta_learner_score.predict(self.vectorize_parameters(parameters_dict_new))}

        #print(f'New anticipated results: {predictions_new}')
        #print(f'The shape of predictions is: {predictions_current.shape}, {predictions_new.shape}')
        #anticipated_change_score, anticipated_change_fit_time = (predictions_new - predictions_current).tolist()[0]
        anticipated_change_score    = (predictions_new['score'] - predictions_current['score']).tolist()[0]
        anticipated_change_fit_time = (predictions_new['time' ] - predictions_current['time' ]).tolist()[0]
        #print(f'Anticipated score change is {anticipated_change_score:.4f}')
        return self.evaluate_improvement(score_change = anticipated_change_score, time_change = anticipated_change_fit_time)
#        except ValueError, e:
#            # anticipation model was not applied
#            print('Anticipation model was not executed properly.')
#            return False, None,None 
            
    def evaluate_improvement(self, new_score = None, new_fit_time = None, 
                             score_change    = None, time_change  = None):
        if new_fit_time is None: new_fit_time = self.best_model_mean_fit_time
        if new_score    is None: new_score    = self.best_score
        
        sign                         = lambda x: math.copysign(1, x)
        score_improvement            = (score_change or (new_score  - self.best_score                 )) * sign(self.step_threshold)
        time_improvement             = time_change   or (self.best_model_mean_fit_time - new_fit_time)
        model_improved_materially    = (score_improvement - time_improvement * self.tradeoff_scr_per_sec> abs(self.step_threshold)) & (new_fit_time < (self.max_fit_time or math.inf))
        
        #if (model_improved_materially == True or model_improved_materially is None) and \
        #   (self.max_fit_time is None         or new_fit_time < self.max_fit_time) 
           
        return model_improved_materially, score_improvement, time_improvement
