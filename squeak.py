import time
from collections import namedtuple
from datetime import timedelta

import numpy as np
import scipy
from networkx import DiGraph, dfs_successors, balanced_tree, path_graph
from sklearn.metrics import pairwise_kernels


#####################################################
# Basic data structures                             #
#####################################################

class SampleDict(object):
    """A collection of samples, together with their reweighting constants
       and other information necessary to build a nystrom reconstruction

    Attributes
    ----------
    X : array, shape (q, n_features)
        Samples included in the dictionary. Also known as inducing points, support vectors or anchor points.
    probs : array, shape (q,)
        The inclusion probability of each sample, used for coin flipping and reweighting.
    q_i : array, shape (q,)
        The number of copies of each sample, used for coin flipping and reweighting. When q_i[i] reaches 0, the sample
        is removed from the dictionary.
    qbar : float
        The number of copies used to initialize the sequential Binomial sampling process and for reweighting.
    gamma : float
        The regularization parameter used to compute the Ridge Leverage Scores.
    vareps : float
        The accuracy parameter used to compute the Ridge Leverage Scores.
    """

    def __init__(self, X, probs, q_i, qbar, gamma, vareps):
        self.X = X
        self.probs = probs
        self.q_i = q_i
        self.gamma = gamma
        self.vareps = vareps
        self.qbar = qbar


class MergePromise(namedtuple('MergePromise', ('is_finished', 'result'))):
    """A simple wrapper for an arbitrary promise (sometimes called a "future" or "async call") implementation.

    Attributes
    ----------
    is_finished : bool
        whether the promise has finished running or not.
    result : any
        The promised result of the computation, containing the result of a dictionary merge.
    """
    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __init__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()


def get_rpc_invoker(backend='redis', url='localhost', port=6379, async_eval=True):
    """Returns an invoker that satisfies the signature rpc_invoker(merge_function, **merge_args) -> MergePromise.
    The actual implementation depends on the backend and can be replaced if necessary.
    The currently implemented backends are based on dask.distributed or redis and redis-queue.
    If the url is 'localhost' and async_eval is False, it will return a local executor without the need of installing
    any backend.

    Parameters
    ----------
    backend : string (optional, default='redis')
        Backend to be used for computation.
    url : string (optional, default='localhost')
        Url of the redis server.
    port : int (optional, default=6379)
        TCP port of the redis server.
    async_eval : bool (optional, default=True)
        Whether the execution should be delayed or not. Set to False to force sequential execution or for
        debugging purposes.

    Returns
    -------
    call_rpc_function : callable
        A callable object that will execute the merge using an arbitrary backend.
    """

    # run local experiment
    if url == 'localhost' and not async_eval:

        def call_rpc_function(func, *args):
            return MergePromise(is_finished=True, result=func(*args))

        return call_rpc_function

    else:
        if backend == 'redis':
            from redis import Redis
            from rq import Queue

            task_queue = Queue(
                connection=Redis(url, port, password="yourpasswordhere"),
                async=async_eval)

            def call_rpc_function(func, *args):
                return task_queue.enqueue(func, *args, timeout=86400)

            return call_rpc_function
        elif backend == 'dask':
            from dask.distributed import Client
            # TODO: for now all workers and the scheduler do not use autentication, this means anyone can submit
            # almost arbitrary code to run. Add TLS (but user needs to know how to use it)
            client = Client("{}:{}".format(url, port))

            def call_rpc_function(func, *args):
                class DaskPromise(object):
                    def __init__(self, dask_promise):
                        self.dask_promise = dask_promise

                    @property
                    def is_finished(self):
                        return self.dask_promise.done()

                    @property
                    def result(self):
                        try:
                            return self.dask_promise.result(timeout=10)
                        except TimeoutError:
                            return None

                return DaskPromise(client.submit(func, *args))

            return call_rpc_function

        else:
            raise NotImplementedError



def dict_merge(d_left, d_right, kernel_options, random_state):
    """Merges two dictionaries, and performs a rejection sampling step according to the new RLS
    to discard redundant samples.

    Parameters
    ----------
    d_left : SampleDict
        The first dictionary to merge (from left descendent).
    d_right : SampleDict
        The second dictionary to merge (from right descendent).
    kernel_options : mapping of string to any
        Dictionary containing the kernel function used as a similarity, with all associated parameters (keyword arguments).
    random_state : RandomState !!MODIFIED!!
        RandomState instance used as the random number generator

    Returns
    -------
    d_top : SampleDict
        The result of the merge.
    """

    # create a temporary dictionary as the concatenation of the two inputs
    d_top = SampleDict(X=np.concatenate((d_left.X, d_right.X)),
                       probs=np.concatenate((d_left.probs, d_right.probs)),
                       q_i=np.concatenate((d_left.q_i, d_right.q_i)), qbar=d_left.qbar, gamma=d_left.gamma,
                       vareps=d_left.vareps)

    # estimate RLS
    tau = estimate_tau(d_top, kernel_options)

    # check for numerical problems
    assert np.all(tau > np.finfo(float).eps)

    # reject sample
    for s in range(len(d_top.q_i)):
        # remember that to make the sampling well-defined, we need to take this minimum
        # it is always a legitimate operation, since increase in RLS can only be due to approximation error
        new_prob = np.minimum(tau[s], d_top.probs[s])
        d_top.q_i[s] = random_state.binomial(d_top.q_i[s], new_prob / d_top.probs[s])
        d_top.probs[s] = new_prob

    # non-zero returns a 1-tuple, extract the only elements
    survived_samples = np.nonzero(d_top.q_i)[0]

    d_top.X = d_top.X[survived_samples]
    d_top.probs = d_top.probs[survived_samples]
    d_top.q_i = d_top.q_i[survived_samples]

    q = d_top.q_i.shape[0]

    assert d_top.X.shape == (q, d_left.X.shape[1])
    assert d_top.q_i.shape == (q,)  # well, duh
    assert d_top.probs.shape == (q,)

    assert (d_top.q_i != 0).all()

    return d_top


def estimate_tau(d_top, kernel_options):
    """Given a sample dictionary, estimates the gamma-Ridge Leverage Scores (RLS) tau of all samples in the dictionary
	to vareps precision.

    Parameters
    ----------
    d_top : SampleDict
        The dictionary whose taus need to be estimated.
    kernel_options : mapping of string to any
        Dictionary containing the kernel function used as a similarity, with all associated parameters (keyword arguments).

    Returns
    -------
    tau : array shape (q,)
        The estimated RLS.
    """

    # just aliases for conciseness
    gamma = d_top.gamma
    vareps = d_top.vareps
    qbar = d_top.qbar

    q = len(d_top.q_i)

    # double check we did not end up dropping all samples because of a too large gamma
    assert q > 0

    # construct kernel matrix between samples in the dictionary
    K = pairwise_kernels(d_top.X, **kernel_options, filter_params=True)

    assert K.shape == (q, q)

    # compute the dictionary weights
    s = (np.sqrt(d_top.q_i) / np.sqrt(d_top.probs)) / np.sqrt(float(qbar))

    # test for numerical errors
    assert (s > np.finfo(float).eps).all()

    # the multiplication automatically creates a copy
    # TODO: could delete K here to free up some memory
    SKS = s[:, np.newaxis] * K * s

    # this avoids creating an additional copy
    np.fill_diagonal(SKS, SKS.diagonal() + gamma)

    # this is a different reformulation of the estimator reported in the paper
    # it is strictly equivalent to the one in the paper ONLY for samples in the dictionary, but we only estimate
    # RLS for samples in the dictionary
    # on the plus side it is more efficient and simpler
    # TODO: check if the inv is the most efficient and numerically stable choice, replace with e.g. pinv(), svd() or solve()
    tau = (1. - 2 * vareps) * np.power(
        np.sqrt(np.ones(q) - gamma * np.diag(scipy.linalg.inv(SKS, overwrite_a=True))) / s, 2)

    return tau


def visit_merge_tree(X, merge_tree, root, rpc_invoker, exp_options, random_state):
    """Visits the merge tree, computing the necessary merges. All squeak variants simply feed a different tree to this
    routine. Leaves in the input tree must possess a 'leaf_assigned_sample' field containing an np.array of indices,
    indicating which samples in the dataset are assigned to that leaf. For all interior nodes, visit_merge_tree will:
    (1) collect the dictionaries contained in the two descendant
    (2) invoke the dict_merge function on the two dictionaries ,using the rpc_invoker to execute it asynchronously
    (3) generate a new dictionary (wrapped in a MergePromise) and assign it to the ancestor node in a 'merge_promise' field
    The computation proceeds from the leaves to the root. Once it completes, the 'merge_promise' field
    in the root contains a dictionary (again wrapped in a Mergepromise) that well approximates the whole dataset.

    Required fields in exp_options to run the algorithm are:
    'qbar': The number of copies used to initialize the sequential Binomial sampling process and for reweighting
    'gamma': The regularization parameter used to compute the Ridge Leverage Scores.
    'vareps': The accuracy parameter used to compute the Ridge Leverage Scores.
    'max_dict_size': The maximum number of samples to store in the dictionary, if a dictionary exceeds this threshold
        the algorithm aborts
    'kernel_options': Dictionary containing the kernel function used as a similarity, with all associated
        parameters (keyword arguments)

    Parameters
    ----------
    X : array, shape (q, n_features)
        Input samples.
    merge_tree : networkx.DiGraph !!MODIFIED!!
        The binary tree guiding the dictionary merges.
    root : int
        Index of the root node.
    rpc_invoker : callable
        An invoker that satisfies the signature rpc_invoker(dict_merge , [merge args]) -> MergePromise.
    exp_options : mapping of string to any
        Dictionary containing the experiment options.
    random_state : RandomState !!MODIFIED!!
        RandomState instance used as the random number generator

    Returns
    -------
    MergePromise containing a dictionary that well approximates the whole dataset.
    """

    leaves = [leaf for leaf in merge_tree.nodes_iter() if merge_tree.out_degree(leaf) == 0]

    # for each leaf, wrap the assigned samples into a completed MergePromise, with a dictionary containing all samples
    # with multiplicity qbar and probability 1. These initialization dictionaries are guaranteed to be accurate
    # since they simply store all samples.
    for leaf in leaves:
        n_i = merge_tree.node[leaf]['leaf_assigned_samples'].shape[0]
        merge_tree.node[leaf]['merge_promise'] = MergePromise(is_finished=True,
                                                              result=SampleDict(
                                                                  X=X[merge_tree.node[leaf]['leaf_assigned_samples'],
                                                                    :], probs=np.ones(n_i),
                                                                  q_i=np.ones(n_i, dtype=int) * (
                                                                      int(np.round(exp_options['qbar']))),
                                                                  qbar=int(np.round(exp_options['qbar'])),
                                                                  gamma=float(exp_options['gamma']),
                                                                  vareps=float(exp_options['vareps'])))

    # total number of merges we need to do is the number of interior nodes, which is number of leaves - 1 since this
    # is a binary tree
    merge_total = len(leaves) - 1

    # we track the runtime of the algorithm
    start_merging_time = time.time()

    # infinite loop, we will break out of it once the root's promise is completed
    while True:

        # declare the flag
        did_something_flag = False

        # TODO: trasversing the merge_tree is cheap compared to running the algorithm, so we do not really optimize for now
        for node, successors in dfs_successors(merge_tree, root).items():

            # we set the flag before all the checks
            did_something_flag = False

            # double check that the tree is not malformed
            assert len(successors) == 2 or len(successors) == 0

            # we do not process leaves
            if len(successors) == 0:
                continue

            # we already scheduled this merge, nothing to do on this node
            if 'merge_promise' in merge_tree.node[node]:
                continue

            # extract the descendent nodes
            l_node = merge_tree.node[successors[0]]
            r_node = merge_tree.node[successors[1]]

            # if one of the descendent does not contain a promise, it is not ready and nothing to do on this node
            if 'merge_promise' not in l_node or 'merge_promise' not in r_node:
                continue

            # if one of the descendent's promise is not done, it is not ready and nothing to do on this node
            if not l_node['merge_promise'].is_finished or not r_node['merge_promise'].is_finished:
                continue

            # we cannot pass directly random_state to the rpc_invoker call, since the rpc could be executed on a remote
            # machine where modification to random_state do not propagate and compromise reproducibility
            # instead, we send over a reproducible random seed and expect the remote system to use it to initialize
            # a local reproducible rng
            merge_random_state = np.random.RandomState(
                random_state.randint(np.iinfo(np.uint32).min + 10,
                                     high=np.iinfo(np.uint32).max - 10))

            # unwrap the descendent's results (leaf nodes were wrapped too)
            l_dict = l_node['merge_promise'].result
            r_dict = r_node['merge_promise'].result

            # if the combined budget size exceed max_dict_size, terminate since we do not want to exceed the machine
            # memory
            assert len(l_dict.q_i) + len(r_dict.q_i) <= 2 * exp_options['max_dict_size']

            # schedule the merge, and assign it to the ancestor
            merge_tree.node[node]['merge_promise'] = rpc_invoker(dict_merge,
                                                                 l_dict,
                                                                 r_dict,
                                                                 exp_options['kernel_options'],
                                                                 merge_random_state)
            # we did something, let the printing function know
            did_something_flag = True

        # we did something, time to log some statistics
        if did_something_flag:
            # how many merges are currently going on
            merge_running = 0
            for node_count_merges in merge_tree.nodes_iter():
                if ('merge_promise' in merge_tree.node[node_count_merges]
                        and not merge_tree.node[node_count_merges]['merge_promise'].is_finished):
                    merge_running = merge_running + 1

            # how many are left before the end
            merge_remaining = 0
            for node_count_merges in merge_tree.nodes_iter():
                if 'merge_promise' not in merge_tree.node[node_count_merges]:
                    merge_remaining = merge_remaining + 1

            print(
                "merge remaining {}/{},".format(merge_remaining, merge_total)
                + "merge currently running {},".format(merge_running)
                + "time elapsed {},".format(str(timedelta(seconds=time.time() - start_merging_time)))
                + "last merge {: 5d} + {: 5d} -> {: 5d}".format(successors[0], successors[1], node)
            )
        else:
            # if we did nothing, sleep a bit to avoid wasting CPU cycles. the user can also enjoy half second of rest
            time.sleep(0.5)

        # we are done, bail out
        if 'merge_promise' in merge_tree.node[root] and merge_tree.node[root]['merge_promise'].is_finished:
            break

    return merge_tree.node[root]['merge_promise']


def squeak(X, exp_options, random_state=None):
    """Invokes visit_merge_tree with a completely unbalanced (sequential) tree. See visit_merge_tree for more details

    Parameters
    ----------
    X : array, shape (q, n_features)
        Input samples.
    exp_options : mapping of string to any
        Dictionary containing the experiment options, see visit_merge_tree and get_rpc_invoker.
    random_state : RandomState (optional, default=None)
        RandomState instance used as the random number generator. If None, gets automatically initialized to
        a fixed number.

    Returns
    -------
    MergePromise containing a dictionary that well approximates the whole dataset.
    """
    if not random_state:
        random_state = np.random.RandomState(42)

    n = float(X.shape[0])  # number of samples
    m = exp_options['max_dict_size']
    k = int(np.ceil(n / m))  # number of initial chunks

    # randomly assign samples to chunks
    perm_idx = np.array_split(random_state.permutation(int(n)), k)

    # to construct a fully unbalanced tree with k leaves, first we create a line graph with k nodes
    merge_tree = path_graph(k, create_using=DiGraph())
    root = min(merge_tree.nodes_iter())

    # the end of the chain is a leaf, so we assign it the last chunk
    end_of_chain = max(merge_tree.nodes_iter())
    merge_tree.node[end_of_chain]['leaf_assigned_samples'] = perm_idx[k - 1]

    # for all the other chunks we look for a (non-leaf) node that does not have samples assigned
    # and also does not have a leaf attached, and attach a leaf with the associated chunk
    for i in range(k - 1):
        for node, successors in dfs_successors(merge_tree, root).items():
            if 'leaf_assigned_samples' not in merge_tree.node[node] and len(successors) < 2:
                merge_tree.add_node(end_of_chain + i + 1, leaf_assigned_samples=perm_idx[i])
                merge_tree.add_edge(node, end_of_chain + i + 1)
                # after we add the node, we have to break, we do not want to add the i-th chunk too many times
                break

    rpc_invoker = get_rpc_invoker(**exp_options['rpc_invoker_options'])
    root_dict_promise = visit_merge_tree(X, merge_tree, root, rpc_invoker, exp_options, random_state)

    return root_dict_promise


def disqueak(X, exp_options, random_state=None):
    """Invokes visit_merge_tree with a completely balanced (fully parallel) tree. See visit_merge_tree for more details

    Parameters
    ----------
    X : array, shape (q, n_features)
        Input samples.
    exp_options : mapping of string to any
        Dictionary containing the experiment options, see visit_merge_tree and get_rpc_invoker.
    random_state : RandomState (optional, default=None)
        RandomState instance used as the random number generator. If None, gets automatically initialized to
        a fixed number.

    Returns
    -------
    MergePromise containing a dictionary that well approximates the whole dataset.
    """
    if not random_state:
        random_state = np.random.RandomState(42)

    n = float(X.shape[0])  # number of samples
    m = exp_options['max_dict_size']
    k = int(np.ceil(n / m))  # number of initial chunks

    if k % 2:
        k += 1  # make it even

    # randomly assign samples to chunks
    perm_idx = np.array_split(random_state.permutation(int(n)), k)

    # we need to fit k chunks in a tree that:
    # (1) is balanced, each node has exactly two descendant or zero (leaf)
    # (2) holds data only in leaves
    # (3) has a difference between the minimum and maximum path in the tree smaller or equal than 1 (in worst case, only
    #       one round difference between best and worst path)
    # To obtain this we compute h, the closest power of two smaller than the number of leaves

    h = int(np.floor(np.log2(k)))

    # create a complete binary tree of depth h

    merge_tree = balanced_tree(2, h, create_using=DiGraph())

    # compute diff, the difference between the number of leaves we want to insert k and the available leaves 2**h

    diff = k - 2 ** h

    # find all leaves
    leaves = [leaf for leaf in merge_tree.nodes_iter() if merge_tree.out_degree(leaf) == 0]

    # then we assign all chunks to the leaves. since we reserved 2**h leaves instead of k, for the first diff
    # rounds we simply add two descendent. every time we do this, we reduce diff by one (since we processed two chunks
    # instead of one) until we have no more difference and we can directly assign to the leaves
    k_i = 0
    node_i = max(merge_tree.nodes_iter())
    for leaf in leaves:
        if diff == 0:
            # we are done with extra chunks, simply assign
            merge_tree.node[leaf]['leaf_assigned_samples'] = perm_idx[k_i]
            k_i = k_i + 1
        else:
            # we do not have enough space to directly assign, add two descendants to the current node and assign
            # to those instead
            merge_tree.add_node(node_i + 1, leaf_assigned_samples=perm_idx[k_i])
            merge_tree.add_node(node_i + 2, leaf_assigned_samples=perm_idx[k_i + 1])
            merge_tree.add_edge(leaf, node_i + 1)
            merge_tree.add_edge(leaf, node_i + 2)
            node_i = node_i + 2
            k_i = k_i + 2

            # the number of leftover leaves get closer to the right one
            diff = diff - 1

    rpc_invoker = get_rpc_invoker(**exp_options['rpc_invoker_options'])
    root_dict_promise = visit_merge_tree(X, merge_tree, 0, rpc_invoker, exp_options, random_state)

    return root_dict_promise
