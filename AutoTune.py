# implementation of step-by-step parameter search
# steps are generated randomly

import copy, random, math
import numpy as np
import pandas as pd
import gc
import sklearn

from sklearn.model_selection import GridSearchCV

# decided not to inherit it for now from GridSearchCV for now,
# instead creating an instance of this class internally
class StepSearchCV():    
    def __init__(
        self,
        estimator        = None,                       # reference to a base estimator, or pipeline of estimators
        parameters_start = None,                       # initial parameters
        prior_results    = None,                       # optional dictionary with prior results, to be extended with new results of the same type
        prior_best_score = None,                       # optional best model score, as calibrated on prior results
        step_rules       = None,                       # optional dictionary containing two scales - for random additive and multiplicative transformations, see default below
        step_veto_rule   = None,                       # optional function of current and new value that returns True if the step is not interesting, see default below
        steps            = math.inf,                   # number of randopm steps requested, could be over-written during fit() method call
        step_threshold   = 0.001,                      # set to positive if higher model performance score is considered better
        scoring          = None,                       # if None the search would rely on estimator model class internal method score(self, X, y)
        random_state     = 7,                          # not currently implemented 
        save_location    = '../output/cv_results.csv', # by default exporting to the parent folder, as assuming current directory would be 'code'
        cv_folds         = 2,                          # number of folds (K) to be used for K-fold cross-validation 
        refit            = False,                      # whether to re-fit each model on the combined set 
        reset_session    = True,                       # whether to re-set each Keras session after every iteration
        extending_lists  = True                        # whethyr list and tuple parameters could be extended
        ):
        
        # initialization

        # TODO: consider how to make results reproducible - with the help of estimator object
        
        # np.random.seed(random_state)
        # random.seed(random_state)

        # adjust default input
        if step_rules     == None: step_rules = {'multiply_scale': 1.0, 'add_scale': 0}
        if step_veto_rule == None: step_veto_rule = lambda x,y: (x >=0.95*y) and (x <=1.05*y)
        
            
        # save settings provided with constructor internally
        self.estimator         = estimator
        self.results           = None        
        self.best_score        = prior_best_score
        self.parameters_best   = parameters_start
        self.step_rules        = step_rules
        self.steps_requested   = steps
        self.step_veto_rule    = step_veto_rule
        self.step_threshold    = step_threshold
        self.save_location     = save_location
        self.scoring           = scoring
        self.cv_folds          = cv_folds
        self.refit             = refit
        self.reset_session     = reset_session
        self.extending_lists   = extending_lists
        
        #initialize grid search object intended to save search results
        self.searcher          = None

        if prior_results is not None:
            self.results = copy.deepcopy(prior_results)
        return 

    def adjust_single_parameter(self, current_value):
        new_value = current_value        
        if type(current_value) is bool: new_value = not(current_value)
        else:    
            while (self.step_veto_rule(current_value, new_value)):
                step_direction                       = random.choice([1,-1])
                step_multiply                        = math.exp(step_direction * abs(np.random.normal(scale = self.step_rules['multiply_scale']))) 
                step_add                             =          step_direction * abs(np.random.normal(scale = self.step_rules['add_scale']     ))
                new_value                            = type(current_value)(step_multiply * current_value + step_add)
        return new_value
        
    def pick_parameter_step(self):
        # note these steps could become more intelligent later as the system learns what works best
        # may want to review what paremeters were already tried 
        # randomly generate next step to explore
        current_parameters      = self.parameters_best
        parameters_under_review = copy.deepcopy(current_parameters)
        parameter_to_adjust     = random.choice(sorted( [key for key in parameters_under_review.keys()] ))
        current_parameter_value = parameters_under_review[parameter_to_adjust]
        
   
        if type(current_parameter_value) in (float, int, bool):  
            parameters_under_review[parameter_to_adjust] = self.adjust_single_parameter(current_parameter_value)
        elif type(current_parameter_value) in (list, tuple):
            #cast tuples to list in order to support elementwise assignment
            sequence_type      = type(current_parameter_value)
            new_sequence_value = list(current_parameter_value)            
            
            # Parameter is a sequence of elements, so we would pick and adjust one of the list elements randomly
            # Alternatively, we could also augment list with additional entry
            current_parameter_sequence_length = len(current_parameter_value)
            parameter_index_to_adjust         = random.choice(np.arange(current_parameter_sequence_length + int(self.extending_lists))) #add 1 for extending lists
            
            if parameter_index_to_adjust < current_parameter_sequence_length:
                # adjust selected paramter, 
                
                current_sequence_element_value = current_parameter_value[parameter_index_to_adjust]
                if type(current_sequence_element_value) in (float, int, bool):  
                    # ready to adjust selected element of the sequence
                    new_sequence_element_value                    = self.adjust_single_parameter(current_sequence_element_value)                    
                    new_sequence_value[parameter_index_to_adjust] = new_sequence_element_value
            else:
                #augment the list with one more entry, equal to minimum entry from the prior list
                new_sequence_element_value    = min(current_parameter_value)
                new_sequence_value.append(new_sequence_element_value)
            
            parameters_under_review[parameter_to_adjust] = sequence_type(new_sequence_value) #cast back to tuple if necessary            
        
        #TODO: also return parameter changed, index and direction
        return parameters_under_review 
    
    def fit(self, X, y=None, groups=None, export_improvements_only = True, steps_completed = 0, more_steps = None, **fit_params):

        if more_steps is not None and type(more_steps) is int: self.steps_requested = more_steps + steps_completed
        
        # iterate until number of steps left drops to zero, or keyboard interrupt
        step                        = steps_completed
        parameters_under_review     = self.parameters_best
        model_improved_materially   = True
        
        while step < self.steps_requested:
            print ("Evaluating: ",  parameters_under_review)
        
            # evaluate selected models and save calibration results internally
            # evaluate model on the parameters selected to be next
            search_grid_wrapper = {key:[item] for key, item in parameters_under_review.items()}
            
            self.searcher = GridSearchCV(
                self.estimator,
                search_grid_wrapper,
                scoring            = self.scoring,
                return_train_score = False,
                cv                 = self.cv_folds,
                refit              = self.refit
                )
            
            try:
                self.searcher.fit(X, y, groups)
                
                # save test results internally
                # TODO: save information of what parameter changed relative to benchmark model (current best)
                if self.results is None: 
                    self.results                 = self.searcher.cv_results_               
                    self.results['iteration']    = np.array(1   , dtype = 'int')
                    self.results['improved' ]    = np.array(True, dtype = 'bool')
                    #self.results['changes'  ]    = np.array(''  , dtype = 'str' )
                    self.best_score              = self.searcher.best_score_
                    #if self.refit == True:
                    #self.estimator           = self.searcher.best_estimator_
                    
                else:
                    # augment cross-validation results dictionary with new test                
                    self.results['iteration']    = np.append(self.results['iteration'],step + 1)
                    # self.results['changes'  ]    = np.append(self.results['changes'  ],'Adjusted' + ) TODO: complete
                    for key, item in self.searcher.cv_results_.items():
                        self.results[key]        = np.append(self.results[key],item) #[0]

                    # decide whether the newly calibrated model is worthy 
                    sign                         = lambda x: math.copysign(1, x)
                    score_improvement            = (self.searcher.best_score_ - self.best_score) * sign(self.step_threshold)
                    model_improved_materially    = score_improvement > abs(self.step_threshold)
                    
                    if model_improved_materially == True or model_improved_materially is None:
                        self.best_score       = self.searcher.best_score_ 
                        #self.estimator        = self.searcher.best_estimator_
                        self.parameters_best  = parameters_under_review
                        print ("Imroved score by %.3f: "% score_improvement , parameters_under_review)

                    self.results['improved'] = np.append(self.results['improved'], model_improved_materially)
                    #print(self.results)

                # Export combined estimation results to a file, if warranted
                if (export_improvements_only == False) or (model_improved_materially == True):
                    self.results_df = pd.DataFrame.from_dict(self.results)
                    self.results_df.to_csv(self.save_location, index = False)
            
            #except ResourceExhaustedError as e:
            #    print('Exhausted memory resources while trying to fit the new model in memory... \n', e)
            except MemoryError as e:
                print('Out of RAM while trying to fit the new model. \n', e)


            # clean up memory resources after modeling completes
            if self.reset_session == True:
                from keras import backend as K
                K.clear_session()            
                self.searcher = None
                gc.collect()

            # move on to the next step, picking a random step within hyper-parameter space
            step += 1
            parameters_under_review     = self.pick_parameter_step()
            
        return self.parameters_best #self.estimator, self.best_estimator
