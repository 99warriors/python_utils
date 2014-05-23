
#import python_utils.caching as caching

import python_utils.python_utils.decorators as decorators
import numpy as np
import functools
import python_utils.python_utils.exceptions
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import math
import itertools
import string
import time
import sys
import multiprocessing
import types


def get_callable_name(f):
    if isinstance(f, functools.partial):
        return get_callable_name(f.func)
    elif isinstance(f, types.FunctionType):
        return f.__name__
    else:
        try:
            return f.__class__.__name
        except:
            return repr(f)


get_shallow_rep = get_callable_name


def get_for_json(o):
    """
    first see if a rich representation is possible
    """
    if isinstance(o, functools.partial):
        return [\
            get_for_json(o.func),\
                [get_for_json(a) for a in o.args],\
#                {k: get_for_json(v) for (k, v) in o.keywords.iteritems() if k[0] != '_'}\
                ]
    try:
        d = o.__dict__
    except AttributeError:
        return get_shallow_rep(o)
    else:
        return [\
            get_shallow_rep(o),\
                {k:get_for_json(v) for (k, v) in d.iteritems() if k[0] != '_'}\
                ]


def json_shortener(s):
    def keep(line):
        import re
        return len(re.findall('[^\[\]\{\},]', line.strip())) > 0
    return string.join(filter(keep, string.split(s, sep='\n')), sep='\n')


def exception_catcher_decorator_helper(f, val, exception_tuple, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except exception_tuple:
        return val


class log_output_fxn_decorator(decorators.fxn_decorator):
    """
    creates log folder, and names log file based on pid
    """

    def __init__(self, log_folder):
        self.log_folder = log_folder
        import os
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

    def __call__(self, f):

        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            import os
            log_file = '%s/%s' % (self.log_folder, str(os.getpid()))
            print log_file, 'GGGGGGGGGGGGGGGGGGGGGGGGG'
            #sys.stdout = open(log_file, 'w')
            return f(*args, **kwargs)

        return wrapped_f


def call_with_logging(log_folder, f, arg):
    import os
    log_file = '%s/%s' % (log_folder, str(os.getpid()))
    import sys
    sys.stdout = open(log_file, 'w')
    sys.stderr = sys.stdout
    return f(arg)


def parallel_map(num_processes, f, iterable):
    """                                                                                                                                                                                                    
    make a                                                                                                                                                                                                 
    """
    import multiprocessing
    results = multiprocessing.Manager().list()
    iterable_queue = multiprocessing.Queue()

    def worker(_iterable_queue, _f, results_queue):
        for arg in iter(_iterable_queue.get, None):
            results_queue.append(_f(arg))

    for x in iterable:
        iterable_queue.put(x)

    for i in xrange(num_processes):
        iterable_queue.put(None)

    workers = []

    for i in xrange(num_processes):
        p = multiprocessing.Process(target=worker, args=(iterable_queue, f, results))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    return [x for x in results]


def logged_sync_map(log_folder, mapper, f, iterable):
#    return map(logged_f, iterable)
    import os
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
#    pool = multiprocessing.Pool(num_procs)
    return mapper(functools.partial(call_with_logging, log_folder, f), iterable)
    return pool.map(functools.partial(call_with_logging, log_folder, f), iterable)



class exception_catcher_fxn_decorator(decorators.fxn_decorator):
    """
    wraps a function so that if exception is raised, returns specified object
    """
    def __init__(self, val, exception_tuple):
        self.val, self.exception_tuple = val, exception_tuple

    def __call__(self, f):

        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            return exception_catcher_decorator_helper(f, self.val, self.exception_tuple, *args, **kwargs)

        return wrapped_f


class exception_catcher_decorated_method(decorators.decorated_method):

    def __init__(self, f, val, exception_tuple):
        self.f, self.val, self.exception_tuple = f, val, val, exception_tuple

    def __call__(self, inst, *args, **kwargs):
        return exception_catcher_decorator_helper(functools.partial(self.f, inst), self.val, self.exception_tuple, *args, **kwargs)


class exception_catcher_method_decorator(decorators.method_decorator):

    def __init__(self, val, exception_tuple):
        self.val, self.exception_tuple = val, exception_tuple

    def __call__(self, f):
        return exception_catcher_decorated_method(f, self.val, self.exception_tuple)


class raise_exception_fxn_decorator(decorators.fxn_decorator):
    """
    decorates the function so that it always raises a TooLazyToComputeException.  this is useful if computing a function is expensive and i'm ok with not having the value for now.
    """
    def __init__(self, which_exception):
        self.which_exception = which_exception

    def __call__(self, f):

        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            raise self.which_exception

        return wrapped_f


class raise_exception_decorated_method(decorators.decorated_method):

    def __init__(self, f, which_exception):
        self.f, self.which_exception = f, which_exception

    def __call__(self, inst, *args, **kwargs):
        raise self.which_exception


class raise_exception_method_decorator(decorators.method_decorator):
    """
    method analogue of raise_exception_fxn_decorator
    """
    def __init__(self, which_exception):
        self.which_exception = which_exception

    def __call__(self, f):
        return raise_exception_decorated_method(f, self.which_exception)

def timeit_decorator_helper(f, display_name, *args, **kwargs):
    start_time = time.time()
    ans = f(*args, **kwargs)
    end_time = time.time()
    print '%s took %.3f seconds' % (display_name, end_time - start_time)
    return ans


class timeit_fxn_decorator(decorators.fxn_decorator):

    def __call__(self, f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            return timeit_decorator_helper(f, get_callable_name(f), *args, **kwargs)
        return wrapped_f

class timeit_decorated_method(decorators.decorated_method):

    def __init__(self, f):
        self.f = f

    def __call__(self, inst, *args, **kwargs):
        display_name = '%s_%s' % (repr(inst), get_callable_name(self.f))
        return timeit_decorator_helper(functools.partial(self.f, inst), display_name, *args, **kwargs)


class timeit_method_decorator(decorators.method_decorator):

    def __call__(self, f):
        return timeit_decorated_method(f)


def date_string_to_year(s):
    return int(string.split(string.split(s, sep=' ')[0], sep='/')[2])


def display_text(s, heading='h2'):
    from IPython.display import display_html
    display_html('<%s>%s</%s>' % (heading, s, heading), raw=True)


def display_fig_inline(fig):
    import StringIO
    from IPython.display import display
    from IPython.display import Image
    output = StringIO.StringIO()
    fig.savefig(output, format='png')
    img = Image(output.getvalue())
    display(img)


def plot_bar_chart(ax, labels, values, offset = 0, width = 0.75, label = None, alpha = 0.5, color = 'red'):
    num = len(labels)
    ax.bar(np.arange(num)+offset, values, label = label, alpha = alpha, color = color, width = width)
    ax.set_xticks(np.arange(num) + 1.0/2)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim((0, num))


def print_decorator(f):

    def wrapped(*args, **kwargs):
        ans = f(*args, **kwargs)
        print ans
        return ans

    return wrapped


def get_powerset_iterator(it):
    """
    it is an iterable (not an iterator), so that you can call __iter__ on it multiple times
    """
    return itertools.chain(*[itertools.combinations(it, l) for l in range(1, len(it)+1)])


def flatten(l_of_l):
    return list(itertools.chain(*[l for l in l_of_l]))


class f(object):
    """
    a callable, nothing more
    """
    @property
    def __name__(self):
        return repr(self)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __repr__(self):
        return '_'

    def __cmp__(self, other):
        return cmp(repr(self), repr(other))


class multiple_composed_f(f):

    def __init__(self, *fs):
        self.fs = fs

    def __repr__(self):
        return 'composed_f(%s)' % string.join([get_callable_name(f) for f in self.fs], sep=',')

    def __call__(self, *args, **kwargs):
        ans = self.fs[-1](*args, **kwargs)
        for f in self.fs[:-1]:
            ans = f(ans)
        return ans


class composed_f(f):

    def __init__(self, f, g, unpack=False):
        self.f, self.g, self.unpack = f, g, unpack
    
    def __repr__(self):
        return 'compose(%s,%s)' % (get_callable_name(f), get_callable_name(g))

    def __call__(self, *args, **kwargs):
        if self.unpack:
            return self.f(*self.g(*args, **kwargs))
        else:
            return self.f(self.g(*args, **kwargs))


class tuple_f(f):

    def __init__(self, *fs):
        self.fs = fs

    def __repr__(self):
        return 'tuple_f(%s)' % string.join([get_callable_name(f) for f in self.fs], sep=',')

    def __call__(self, *args, **kwargs):
        return tuple([f(*args, **kwargs) for f in self.fs])


class hard_code_f(f):

    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val

    def __repr__(self):
        return 'hardcoded_f_%s' % repr(self.val)

class series_f(f):

    def __init__(self, *fs):
        self.fs = fs

    def __repr__(self):
        return 'series_f(%s)' % string.join([get_callable_name(f) for f in self.fs], sep=',')

    def __call__(self, *args, **kwargs):
        
        def to_series(f):
            v = f(*args, **kwargs)
            if isinstance(v, pd.Series):
                return v
            else:
                return pd.Series({repr(f):v})

        ans = pd.concat([to_series(f) for f in self.fs])
        try:
            ans.name = args[0]
        except (AttributeError, IndexError):
            pass

        return ans


class DataIdIterable(object):

    def __iter__(self):
        raise NotImplementedError


class negation_f(f):

    def __init__(self, orig):
        self.orig = orig

    def __call__(self, *args, **kwargs):
        return not self.orig(*args, **kwargs)


class and_f(f):

    def __init__(self, *fs):
        self.fs = fs

    def __call__(self, *args, **kwargs):
        return np.sum([f(*args, **kwargs) for f in self.fs]) == len(self.fs)


class or_f(f):

    def __init__(self, *fs):
        self.fs = fs

    def __call__(self, *args, **kwargs):
        return np.sum([f(*args, **kwargs) for f in self.fs]) > 0


class contains_f(f):

    def __repr__(self):
        return '%s_%s' % (repr(f), repr(self.val))

    def __init__(self, f, val):
        self.f, self.val = f, val

    def __call__(self, data_id):
        return self.f(data_id) in self.val


class categorical_f(f):
    """
    returns a vector of 0 and 1's
    """
    def __init__(self, f, bins, other=True):
        self.f, self._bins = f, bins
        self._contains_fs = [contains_f(f, bin) for bin in bins]
        if other:
            self._contains_fs.append(and_f(*[negation_f(_contains_f) for _contains_f in self.contains_fs]))

    def __repr__(self):
        return '%s_%s' % (repr(self.f), string.join([repr(bin) for bin in self.bins], sep='_'))

    @property
    def bins(self):
        return self._bins

    @property
    def contains_fs(self):
        return self._contains_fs

    def __call__(self, data_id):
        #print [repr(contains_f) for contains_f in self._contains_fs]
        #print pd.Series({repr(contains_f):contains_f(data_id) for contains_f in self._contains_fs})
        #print {repr(contains_f):contains_f(data_id) for contains_f in self._contains_fs}
        #pdb.set_trace()
        return pd.Series({repr(contains_f):contains_f(data_id) for contains_f in self._contains_fs})


class mapping(object):
    """
    represents mapping.  could be many to 1
    """
    def f(self, x):
        raise NotImplementedError

    def preimage(self, y):
        raise NotImplementedError


class bijective_mapping(mapping):

    def f_inv(self, y):
        raise NotImplementedError

    def preimage(self, y):
        return [self.f_inv(y)]


class bijective_mapping_from_list(bijective_mapping):
    """
    maps items of a list to their position in the list
    """
    def __init__(self, vals):
        assert len(vals) == len(set(list))
        self.index_to_val = [val for val in vals]
        self.val_to_index = dict((val,i) for (i, val) in enumerate(self.index_to_val))

    def f(self, index):
        return self.index_to_val[index]

    def f_inv(self, val):
        return self.val_to_index[val]


class int_f_from_categorical_f(f, bijective_mapping):
    """
    returns the coordinate that has a 1
    also defines 1-to-1 mapping from contains_fs (which have string representation) to integers
    """
    def __init__(self, _categorical_f):
        self.categorical_f = _categorical_f

    def __repr__(self):
        return 'int_f_%s' % repr(self.categorical_f)

    def f(self, contains_f):
        return np.where(self.categorical_f.contains_fs == contains_f)[0][0]

    def f_inv(self, index):
        return self.categorical_f.contains_fs[index]

    def __call__(self, data_id):
        v = self.categorical_f(data_id)
        nonzeros = np.nonzero(v)[0]
        try:
            assert len(nonzeros) == 1
        except AssertionError:
            import pdb

        return nonzeros[0]


class feature(object):
    """
    a function that depends jointly on an entire data_id_iterable, as it could involve operations on a list of feature values like normalization.  returns a dataframe
    """
    def __call__(self, data_id_iterable):
        raise NotImplementedError


class raw_feature(feature):
    """
    feature that just applies passed in f to every id in data_id_iterable
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, data_id_iterable):
        l = {id:self.f(id) for id in data_id_iterable}
        # test if the feature is an iterable
        a_val = l.iteritems().next()[1]
        if isinstance(a_val, pd.Series):
            return pd.DataFrame(l).T
        else:
            ans = pd.DataFrame(pd.Series(l, name=str(self.f)))
            return ans


class sklearn_categorical_feature(feature):
    """
    uses sklearn's DictVectorizer
    assume that f returns a string for now
    drops na data
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, data_id_iterable):
        # if there are na's, those become an additional category, so have to get rid of it
        vals = pd.Series({id:self.f(id) for id in data_id_iterable}, name=self.f).dropna()
        from sklearn.feature_extraction import DictVectorizer
        encoder = DictVectorizer()
        X = pd.DataFrame(encoder.fit_transform([{self.f:val} for val in vals]).todense())
        X.index = vals.index
        X.columns = encoder.feature_names_
        return X
        

class processor(object):
    """
    function on entire dataframe.  compose with the feature class to do processing, like normalization 
    """
    def __call__(self, X):
        raise NotImplementedError


class pipeline_wrapped_processor(processor):
    """
    wrapper for sklearn pipeline.  all it does is ensure the output is a dataframe, and set the indices/columns
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, X):
        ans = self.pipeline.fit_transform(X)
        return pd.DataFrame(ans, index = X.index, columns = X.columns)


def get_data(data_id_iterable, feature_list):
    """
    takes in list of features, outputs the data dataframe
    """
    return pd.concat([f(data_id_iterable) for f in feature_list], join='outer', axis=1)


def pystan_traces_to_list_of_dicts(traces):
    """
    point is that each sample of params can be passed directly to function that simulates a sample from the model distribution
    """
    n = traces.iteritems().next()[1].shape[0]
    l = [{} for i in xrange(n)]
    for key, trace in traces.iteritems():
        if key != 'lp__':
            for i in xrange(n):
                if len(trace.shape) == 1:
                    l[i][key] = trace[i]
                elif len(trace.shape) == 2:
                    l[i][key] = pd.Series(trace[i,:])
                elif len(trace.shape) == 3:
                    l[i][key] = pd.DataFrame(trace[i,:,:])
                else:
                    assert False
    return l
    

def merge_pystan_permuted_traces(traces):
    """
    
    """
    keys = iter(traces).next().keys()
    merged = {}
    for key in keys:
        merged[key] = np.concatenate([trace[key] for trace in traces])
    return merged


class mixture_dist(object):

    def __init__(self, pi, dists):
        self.pi, self.dists = pi, dists
        if isinstance(self.pi, float):
            self.pi = [self.pi]
        if len(self.pi) == len(self.dists)-1:
            self.pi.append(1.0 - sum(self.pi))

    def __call__(self, *args, **kwargs):
        import numpy.random
        which = np.where(np.random.multinomial(1, self.pi)==1)[0][0]
        return self.dists[which](*args, **kwargs)


def random_categorical(p):
    """
    p is normalized 
    """
    import numpy.random, math
    p_array = np.array(p)
    return int(math.floor(np.sum(p_array.cumsum() < p_array.sum() * np.random.random())))

        


def get_elt_trace(l, accessor, skip=1):
    import itertools
    iterator = itertools.imap(lambda z: z[1], itertools.ifilter(lambda y: y[0] % skip == 0, enumerate(itertools.imap(accessor, l))))
    return iterator


def plot_trace(l, title=None):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    l = [x for x in l]
    ax.plot(range(len(l)),[x for x in l])
    return fig


class F(object):
    """
    what could F be?
    - a regression function, F(x), so that __call__ returns F(x) or [F(x) for x in xs]
    - a function that returns another function (ie an estimator) so that, g = F(data).  example would be density estimation
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, train_data):
        raise NotImplementedError

    def __call__(self, test_data):
        raise NotImplementedError


class unsupervised_F(F):

    def train(self, *args):
        pass


class hyper_free_F(F):
    """
    
    """
    def __init__(self, F_list, perf_f):
        self.F_list = F_list, perf_f

    def train(self, train_data):
        perfs = []
        for F_trainer in F_trainer_list:
            perfs.append((F_trainer, perf_f(F_trainer, train_data)))
        best_F_trainer = max(perfs, key = lambda x: x[1])[0]
        self.F = best_F_trainer(train_data)

    def __call__(self, test_data):
        return self.F(test_data)

class kd_helper_F(F):
    """
    returns log density
    """
    def __init__(self, bandwidth, kernel, metric):
        self.bandwidth, self.kernel, self.metric = bandwidth, kernel, metric

    def train(self, points):
        from sklearn.neighbors.kde import KernelDensity
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth, metric=self.metric).fit(points)

    def __call__(self, point):
        return self.kde.score(point)


class categorical_joint_distribution_F(F):
    """
    abstract class.  given dataframe where columns are integer vectors, estimates the joint probability over categorical features
    """
    def train(self, train_df):
        pass

    def __call__(self, test_df):
        pass


def is_zero(x):
    return abs(x) < 0.00001


def faster_all(l):
    pass


def multivariate_random_uniform(ranges):
    import numpy.random
    return map(lambda r: numpy.random.uniform(r[0], r[1]), ranges)


class generator_f(f):
    """
    it should be an iterator (not iterable)
    """
    def __init__(self, it):
        self.it = it

    def __call__(self):
        return self.it.next()


class independent_joint_distribution(object):
    """
    if for a coordinate, see a value that is too high, that means it wasn't observed, so has prob 0
    this will cause divide by 0 error if background data is very small (but this shouldn't really happen)
    """
    def __init__(self, margs):
        self.margs = margs

    def get_prob(self, x):
        prob = 0
        for i, x_i in itertools.izip(x):
            prob += self.get_single_coord_prob(i, x_i)
        return prob

    def get_single_coord_prob(self, i, val):
        if val >= len(self.margs[i]):
            return np.log(0.0)
        else:
            return self.margs[i][val]

    def get_restriction(self, restriction):
        from scipy.misc import logsumexp
        prob = 0.0
        for i,s in enumerate(restriction):
            prob += logsumexp([self.get_single_coord_prob(i, idx) for idx in s])
        return prob

    @property
    def shape(self):
        return tuple([len(marg) for marg in self.margs])


class independent_categorical_joint_distribution_F(categorical_joint_distribution_F):

    def __init__(self):
        pass

    def train(self, train_df):
        """
        density estimation is un-supervised (no hyperparameters to choose for now), so this does nothing
        """
        pass

#    @timeit_method_decorator()
    def __call__(self, num_cats, X):
        """
        accepts matrix where each row represents a sample
        """
        margs = [0 for x in xrange(X.shape[1])]
        for i, col in enumerate(X.T):
            #num_vals = np.max(col) + 1
            num_vals = num_cats[i]
            num = len(col)
            try:
                margs[i] = np.array([(np.log(np.sum(col==val)) - np.log(float(num))) for val in xrange(num_vals)])
            except ValueError:
                print [np.sum(col==val) for val in xrange(num_vals)]
                print [(math.log(np.sum(col==val)) - math.log(float(num))) for val in xrange(num_vals)]
                pdb.set_trace()
        return independent_joint_distribution(margs)


class StopIterativeException(Exception):

    def __init__(self, val):
        self.val = val


class cycle_through_coord_iterative_step(object):
    """

    """
    def __init__(self):
        pass

    def __call__(self, f, x, constraints):
        dim = len(x)
        for i in xrange(dim):
            x[i] = f.coord_ascent(x, i, constraints)
        return x


class get_initial_subset_x_random(object):
    """
    to avoid subset for a mode being empty, always have at least 1 row chosen
    """
    def __init__(self, p, seed=0):
        self.p, self.seed = p, seed

    def __call__(self, f, constraints):
        dims = f.x_dims
        ans = [np.array([i for i in xrange(dim) if np.random.uniform() < self.p]) for dim in dims]
        for (i, coord), constraint in zip(enumerate(ans), constraints):
            if len(coord) == 0:
                ans[i] = np.array([0])
            if constraint != None:
                import random
                ans[i] = random.sample(constraint, 1)[0]
                #ans[i] = constraint[np.random.choice(range(len(constraint)))]
        return ans
                

class iterative_argmax_F(unsupervised_F):
    """
    basically the argmax function
    """
    def __init__(self, get_initial_x, iterative_step, max_steps, tol):
        self.get_initial_x, self.iterative_step = get_initial_x, iterative_step
        self.max_steps, self.tol = max_steps, tol

    def __call__(self, f, constraints):
        x = self.get_initial_x(f, constraints)
        f_x = f(x)
        #self.iterative_step.set_f(f)
        for i in xrange(self.max_steps):
            try:
                x_new = self.iterative_step(f, x, constraints)
            except StopIterativeException, e:
                return e.val
            if f(x_new) - f_x > self.tol:
                return x_new
            x = x_new
            f_x = f(x)
        return x

def ndarray_subset(x, subsets):
    #y = x.copy()
    y = x
    m = [slice(None) for l in x.shape]
    for i,s in enumerate(subsets):
        m[i] = s
        try:
            y = y[m]
        except:
            pdb.set_trace()
        m[i] = slice(None)
    return y


def array_restriction(ar, subset):
    """
    sum of elements at specified elements
    """
    sum = 0.0
    for i in subset:
        sum += ar[i]
    return sum


class bin(object):

    def __contains__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def point_rep(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class equals_bin(bin):

    def __repr__(self):
        return 'equals_%s' % repr(self.val)

    def __init__(self, val):
        self.val = val

    def __contains__(self, item):
        #print self.val, item, item == self.val
        return item == self.val

    def __len__(self):
        return 1.0

    def point_rep(self):
        return self.val

    def __hash__(self):
        return self.val.__hash__()


class everything_bin(bin):

    def __repr__(self):
        return 'everything'

    def __contains__(self, item):
        return True

    def __len__(self):
        return 1.0

    def __hash__(self):
        return hash(0)


class range_bin(bin):

    def __repr__(self):
        return 'range_%s_%s' % (repr(self.low), repr(self.high))

    def __init__(self, low, high):
        self.low, self.high = low, high

    def __len__(self):
        return self.high - self.low

    def point_rep(self):
        return self.low

    def __contains__(self, other):
        return other < self.high and other >= self.low


class contains_bin(bin):

    def __init__(self, vals):
        self.vals = set(vals)

    def __contains__(self, val):
        return val in self.vals

    def __len__(self):
        return len(self.vals)

    def __iter__(self):
        return iter(self.vals)

    def __repr__(self):
        return 'contains_%s' % repr(self.vals)


class product_bin(bin):

    def __init__(self, *coord_bins):
        self.coord_bins = coord_bins

    def __contains__(self, v):
        for i, coord_bin in enumerate(self.coord_bins):
            if v[i] not in coord_bin:
                return False
        return True

    def __repr__(self):
        return 'product_bin_%s' % string.join([repr(coord_bin) for coord_bin in self.coord_bins], sep = '_')

    def point_rep(self):
        return tuple([coord_bin.point_rep() for coord_bin in self.coord_bins])

class region(bin):

    def point_rep(self):
        raise NotImplementedError

    def __contains__(self, location):
        raise NotImplementedError


class voronoi_region(region):
    """
    the voronoi region of a point with respect to a set of points
    """
    def __init__(self, coord, kdtree):
        self.coord, self.kdtree = coord, kdtree

    def point_rep(self):
        return self.coord

    def __contains__(self, coord):
        return self.kdtree.data[self.kdtree.query(coord)[1]] == coord

    def __hash__(self):
        return hash(self.coord)

class NoRegionException(Exception):
    pass


class simple_region_list(list):

    def point_to_region_index(self, point):
        for i, region in enumerate(self):
            if point in region:
                return i
        raise NoRegionException
        

class voronoi_region_list(list):

    def __init__(self, voronoi_points):
        from scipy.spatial import KDTree
        self.kdtree = KDTree(voronoi_points)
        voronoi_regions = [voronoi_region(point, self.kdtree) for point in voronoi_points]
        list.__init__(self, voronoi_regions)

    def point_to_region_index(self, point):
        return self.kdtree.query(point)[1]


def latlng_to_xy(lat, lng):
    return lng, lat
    return latlng[1], latlng[0]


class latlng_grid_region(product_bin):

    def __init__(self, lat_low, lat_high, lng_low, lng_high):
        self.lat_low, self.lat_high, self.lng_low, self.lng_high = lat_low, lat_high, lng_low, lng_high
        product_bin.__init__(self, range_bin(lat_low, lat_high), range_bin(lng_low, lng_high))

    def plot(self, ax):
        topleft_x, topleft_y = latlng_to_xy(self.lat_high, self.lng_low)
        topright_x, topright_y = latlng_to_xy(self.lat_high, self.lng_high)
        bottomright_x, bottomright_y = latlng_to_xy(self.lat_low, self.lng_high)
        bottomleft_x, bottomleft_y = latlng_to_xy(self.lat_low, self.lng_low)
        ax.plot([topleft_x, topright_x, bottomright_x, bottomleft_x, topleft_x], [topleft_y, topright_y, bottomright_y, bottomleft_y, topleft_y], color = 'blue')
        return ax

class latlng_grid_region_list(list):

    def __init__(self, num_lat, num_lng, lat_min, lat_max, lng_min, lng_max):
        self.num_lat, self.num_lng = num_lat, num_lng
        self.lat_min, self.lat_max, self.lng_min, self.lng_max = lat_min, lat_max, lng_min, lng_max
        lat_boundaries = np.linspace(self.lat_min, self.lat_max, self.num_lat + 1)
        lng_boundaries = np.linspace(self.lng_min, self.lng_max, self.num_lng + 1)
        regions = [latlng_grid_region(lat_low, lat_high, lng_low, lng_high) for ((lat_low, lat_high), (lng_low, lng_high)) in itertools.product(itertools.izip(lat_boundaries[0:-1], lat_boundaries[1:]), itertools.izip(lng_boundaries[0:-1], lng_boundaries[1:]))]
        list.__init__(self, regions)

    def point_to_region_index(self, latlng):
        lat, lng = latlng[0], latlng[1]
        if lat > self.lat_max or lat < self.lat_min:
            raise NoRegionException
        if lng > self.lng_max or lng < self.lng_min:
            raise NoRegionException
        lat_idx = int(min(int(self.num_lat * (lat - self.lat_min) / (self.lat_max - self.lat_min)), self.num_lat-1))
        lng_idx = int(min(int(self.num_lng * (lng - self.lng_min) / (self.lng_max - self.lng_min)), self.num_lng-1))
        idx = lat_idx * self.num_lng + lng_idx
        return idx
        for i, region in enumerate(self):
            if latlng in region:
                if i != idx:
                    print i, idx, latlng in self[i], latlng in self[idx], latlng, self[i], self[idx]
                    pdb.set_trace()
                    assert False
        return idx

    def coord_to_index(self, i, j):
        return i * self.num_lng + j


class grid_region_box_region_subset_F(F):

    def __init__(self, max_region_width):
        self.max_region_width = max_region_width

    def __call__(self, regions):
        """
        assumes regions has a num_lat, num_lng, and coord_to_index method
        """
        # for each top left corner, add all allowable squares
        return set([tuple([x for x in itertools.starmap(regions.coord_to_index, itertools.product(range(i, min(regions.num_lat, i + width)), range(j, min(regions.num_lng, j + width))))]) for i in range(regions.num_lat) for j in range(regions.num_lng) for width in range(1, self.max_region_width)])



class singleton_location_regions_F(unsupervised_F):

    def __init__(self):
        pass

    def __call__(self, locations):
        return simple_region_list([utils.equals_bin(location) for location in locations])


class latlng_grid_regions_F(unsupervised_F):

    def __init__(self, num_lat, num_lng):
        self.num_lat, self.num_lng = num_lat, num_lng

    def __call__(self, locations):
        """
        locations are (lat,lng) tuples.  this one ignores actual locations though
        """
        lat_min = np.percentile([location[0] for location in locations], 5)
        lat_max = np.percentile([location[0] for location in locations], 95)
        lng_min = np.percentile([location[1] for location in locations], 5)
        lng_max = np.percentile([location[1] for location in locations], 95)
        return latlng_grid_region_list(self.num_lat, self.num_lng, lat_min, lat_max, lng_min, lng_max)
        lat_boundaries = np.linspace(self.lat_min, self.lat_max, self.num_lat)
        lng_boundaries = np.linspace(self.lng_min, self.lng_max, self.num_lng)
        regions = [product_bin(range_bin(lat_low, lat_high), range_bin(lng_low, lng_high)) for ((lat_low, lat_high), (lng_low, lng_high)) in itertools.product(itertools.izip(lat_boundaries[0:-1], lat_boundaries[1:]), itertools.izip(lng_boundaries[0:-1], lng_boundaries[1:]))]
        # TODO simple_region_list is not efficient.  could probably use different container
        return simple_region_list(regions)


class performance_f(object):

    def __call__(self, F_trainer, test_data):
        raise NotImplementedError

class in_sample_performance_f(performance_f):

    def __init__(self, summarizer):
        self.summarizer = summarizer

    def __call__(self, F_trainer, data):
        F = F_trainer(data)
        predictions = F(data)
        return self.summarizer(predictions, data)


class cv_performance_f(performance_f):
    """
    for now, just return the mean performance across folds
    """
    def __init__(self, summarizer, folds_iterator):
       self.summarizer, self.folds_iterator = summarizer, folds_iterator

    def __call__(self, F_trainer, data):
        for train_data, test_data in self.folds_iterator(data):
            F = F_trainer(train_data)
            prediction = F(test_data)
            predictions.append(prediction)
        return self.summarizer(predictions, data)


class scalar_dist_mat_f(f):

    def __init__(self, metric):
        self.horse = dist_mat_f(metric)

    def __call__(self, vals):
        X = np.matrix([[val] for val in vals])
        return self.horse(X)


class dist_mat_f(f):
    """
    just a wrapper for scipy.spatial.distance.pdist
    """
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, X):
        """
        each observation is a row
        """
        from scipy.spatial.distance import pdist
        pdb.set_trace()
        return pdist(X, self.metric)


class scaled_dist_mat_F(F):
    """
    training just figures out scaling.  each row of X is an observation
    """
    def __init__(self, metric):
        self.metric = metric

    def train(self, X):
        # FIX
        self.right_transform = np.diag(pd.DataFrame(X).apply(lambda s: 1.0 / (s.quantile(0.9) - s.quantile(0.1))))

    def __call__(self, X):
        X_scaled = X.dot(self.right_transform)
        from scipy.spatial.distance import pdist
        return pdist(X_scaled, self.metric)


class scaled_scalar_dist_mat_F(F):
    """
    training just figures out scaling.  each row of X is an observation
    """
    def __init__(self, metric):
        self.metric = metric

    def train(self, X):
        X_series = pd.Series(X)
        # FIX
        self.scale = 1.0 / (X_series.quantile(0.9) - X_series.quantile(0.1))


    def __call__(self, X):
        from scipy.spatial.distance import pdist
        X_mat = [[x_i] for x_i in X]
        return pdist(X_mat, self.metric) * self.scale


def get_precision_recall(retrieved_set, relevant_set, total_set):
    precision = float(len(retrieved_set.intersection(relevant_set))) / len(retrieved_set)
    recall = float(len(retrieved_set.intersection(relevant_set))) / len(relevant_set)
    return precision, recall


def get_precision_recall_curve_fig(truths, scores):
    fig, ax = plt.subplots()
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    precision, recall, thresholds = precision_recall_curve(truths, scores)
    auc_value = auc(recall, precision)
    ax.plot(recall, precision)
    ax.set_title('auc: %.2f' % auc_value)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    fig.suptitle('precision recall curve')
    return fig, ax


def get_fpr_tpr(retrieved_set, relevant_set, total_set):
    
    fpr = float(len(total_set.difference(relevant_set).intersection(retrieved_set))) / len(total_set.difference(relevant_set))
    tpr = float(len(retrieved_set.intersection(relevant_set))) / len(relevant_set)
    return fpr, tpr

def get_roc_curve_fig(truths, scores):
    fig, ax = plt.subplots()
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    fpr, tpr, thresholds = roc_curve(truths, scores)
    auc_value = auc(fpr, tpr)
    ax.plot(fpr, tpr)
    ax.set_title('auc: %.2f' % auc_value)
    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')
    fig.suptitle('roc curve')
    return fig, ax

def map_with_args(map_f, iterable):
    return [(item, map_f) for item in iterable]

def call(args, kwargs, f):
    return f(*args, **kwargs)

def apply_fs(fs, *args, **kwargs):
    return map(functools.partial(call, args=args, kwargs=kwargs), fs)

def apply_fs_with_args(fs, *args, **kwargs):
    return map_with_args(functools.partial(call, args=args, kwargs=kwargs), fs)
    
def reduced_f(fs, aggregate_f, *args, **kwargs):
    return composed_f(aggregate_f, functools.partial(apply_f_with_args, args=args, kwargs=kwargs))


def scatter(ax, points, transform = lambda x: x, color = None):
    points = [transform(point) for point in points]
    if color == None:
        ax.scatter([point[0] for point in points], [point[1] for point in points])
    else:
        ax.scatter([point[0] for point in points], [point[1] for point in points], c = color)
    return ax


def argmin(l):
    return min([i,x in enumerate(l)], key = lambda (i,x): x)[1]


def argmax(l):
    return max([i,x in enumerate(l)], key = lambda (i,x): x)[1]


def get_data_feature_counts(data):
    X = pd.DataFrame(map(lambda datum:datum.x, data))
    counts_by_feature = map(lambda (name, col): pd.Series(col.value_counts(), name = name), X.iteritems())
    return counts_by_feature


def plot_data_feature_counts(data):
    X = pd.DataFrame(map(lambda datum:datum.x, data))
    counts_by_feature = map(lambda name, col: pd.Series(col.value_counts(), name = name), X.iteritems())
    fig_axes = []
    for counts in counts_by_feature:
        fig, ax = plt.subplots()
        labels = [str(x) for x in count.index]
        utils.plot_bar_chart(ax, labels, counts)
        fig.suptitle(counts.name)
        fig_axes.append((fig, ax))
    return fig_axes


class parallel_map_with_callback(object):
    """
    calls asynchronous ipython map, then checks results to see if they have finished, and calls callback on results if so
    """
    def __init__(self, view):
        self.view = view

    def __call__(self, f, l, callback):
        self.view.map(f, l)


def map_reduce_f(mapper, map_f, reduce_f, iterable):
    return reduce_f(mapper(map_f, iterable))


def apply(x, f):
    return f(x)


def parallel_get_posterior_f(mapper, get_posterior_fs, data):
    posteriors = mapper(functools.partial(apply, data), get_posterior_fs)
    permuted, unpermuted = zip(*posteriors)
    return merge_pystan_permuted_traces(permuted), list(itertools.chain(*unpermuted))


class remapped_kwargs_f(object):
    """
    keyword_map maps new keywords to old keywords
    """

    def __init__(self, f, keyword_map):
        self.f = f
        self.keyword_map = keyword_map

    def __call__(self, *args, **kwargs):
        return self.f(*args, **dict([(self.keyword_map[key], val) for key in kwargs]))
    


#################
# BELOW IS CRAP #
#################

class val_to_bin_index_mapping(mapping):

    def __init__(self, bins):
        self.bins = bins

    def f(self, val):
        has = np.where([val in bin for bin in self.bins])[0]
        assert len(has) == 1
        return has[0]

    def f_inv(self, index):
        return [val for val in self.bins[index]]

    @property
    def max_index(self):
        return len(self.bins)


class filter_dataframe_column_f(object):
    """
    
    """
    def __init__(self, col_filter_f):
        self.col_filter_f = col_filter_f

    def __call__(self, X):
        return pd.DataFrame({col:s for col,s in X.iteritems() if self.col_filter_f(col, s)})
