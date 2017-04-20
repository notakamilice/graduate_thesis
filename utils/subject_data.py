import os
import time
import warnings
import numpy as np
from matplotlib.pyplot import cm
from sklearn.externals.joblib import Memory
from nilearn._utils.compat import _basestring
from .io_utils import (niigz2nii as do_niigz2nii, 
                       nii2niigz as do_nii2niigz,
                       hard_link, 
                       get_shape,
                       is_4D, 
                       is_3D, 
                       get_basenames, 
                       is_niimg)

mc_tooltip = ("Motion parameters estimated during motion-"
              "correction. If motion is less than half a "
              "voxel, it's generally OK. Moreover, it's "
              "recommended to include these estimated motion "
              "parameters as confounds (nuissance regressors) "
              "in the the GLM.")
segment_acronyms = ("Acronyms: TPM means Tissue Probability Map; GM means "
                   "Grey-Matter;"
                   " WM means White-Matter; CSF means Cerebro-Spinal Fuild")
reg_tooltip = ("The red contours should"
               " match the background image well. Otherwise, something might"
               " have gone wrong. Typically things that can go wrong include: "
               "lesions (missing brain tissue); bad orientation headers; "
               "non-brain tissue in anatomical image, etc. In rare cases, it "
               "might be that the registration algorithm simply didn't "
               "succeed.")
segment_tooltip = ("%s. The TPM contours shoud match the background image "
                   "well. Otherwise, something might have gone wrong. "
                   "Typically things that can go wrong include: "
                   "lesions (missing brain tissue); bad orientation headers; "
                   "non-brain tissue in anatomical image (i.e needs brain "
                   "extraction), etc. In rare cases, it might be that the"
                   " segmentation algorithm simply didn't succeed." % (
                       segment_acronyms))
tsdiffana_tooltips = [
    ("(Squared) differences across sequential volumes. "
     "A large value indicates an artifact that occurred during the "
     "slice acquisition, possibly related to motion."),
    ("Average signal over each volume. A large drop / peak (e.g. 1%)"
     " w.r.t the mean level indicates an artefact. For example, there are "
     " usually large values peaks in the first few slices due to T2 "
     " relaxation effects, and these slices are usually adviced to be "
     "discarded."),
    ("Variance index per slice. Note that acquisition artifacts can be slice"
     "-specific. Look at the data if there is a peak somewhere."),
    ("Scaled variance per slice indicates slices where artifacts occur."
    " A slice/time with large variance should be eyeballed."),
    ("Large variations should be confined to vascular structures "
     "or ventricles. Large variations around the brain indicate"
     " residual motion effects."),
    ("Large variations should be confined to vascular structures or"
     " ventricles. Large variations around the brain indicate (uncorrected)"
     " motion effects.")]


class SubjectData(object):
    """
    Encapsulation for subject data, relative to preprocessing.

    Parameters
    ----------
    func: can be one of the following types:
        ---------------------------------------------------------------
        Type                       | Explanation
        ---------------------------------------------------------------
        string                     | one session, 1 4D image filename
        ---------------------------------------------------------------
        list of strings            | one session, multiple 3D image
                                   | filenames (one per scan)
                                   | OR multiple sessions, multiple 4D
                                   | image filenames (one per session)
        ---------------------------------------------------------------
        list of list of strings    | multiple sessions, one list of
                                   | 3D image filenames (one per scan)
                                   | per session
        ---------------------------------------------------------------

    anat: string
        path to anatomical image

    subject_id: string, optional (default 'sub001')
        subject id

    session_ids: string or list of strings, optional (default None):
        session ids for all sessions (i.e runs)

    output_dir: string, optional (default None)
        output directory for this subject; will be create if doesn't exist

    session_output_dirs: list of strings, optional (default None):
        list of output directories, one for each session

    anat_output_dir: string, optional (default None)
        output directory of anatomical data

    scratch: string, optional (default None)
        root directory for scratch data (temp files, cache, etc.) for this
        subject; thus we can push all scratch data unto a dedicated device

    **kwargs: param-value dict_like
        additional optional parameters

    """

    def __init__(self, func=None, anat=None, subject_id="sub001",
                 session_ids=None, output_dir=None, session_output_dirs=None,
                 anat_output_dir=None, scratch=None, warpable=None, **kwargs):
        if warpable is None:
            warpable = ['anat', 'func'] 
        self.func = func
        self.anat = anat
        self.subject_id = subject_id
        self.session_ids = session_ids
        self.n_sessions = None
        self.output_dir = output_dir
        self.anat_output_dir = anat_output_dir
        self.session_output_dirs = session_output_dirs
        self.warpable = warpable
        self.failed = False
        self.warpable = warpable
        self.nipype_results = {}
        self._set_items(**kwargs)
        self.scratch = output_dir if scratch is None else scratch
        self.anat_scratch_dir = anat_output_dir if scratch is None else scratch
        self.session_scratch_dirs = (session_output_dirs if scratch is None
                                     else [scratch] * len(session_output_dirs))

    def _set_items(self, **kwargs):
        for k, v in list(kwargs.items()):
            setattr(self, k, v)

    # def _delete_orientation(self):
    #     """
    #     Delete orientation metadata. Garbage orientation metadata can lead to
    #     severe mis-registration trouble.

    #     """

    #     # prepare for smart caching
    #     if self.scratch is None:
    #         self.scratch = self.output_dir
    #     cache_dir = os.path.join(self.scratch, 'cache_dir')
    #     if not os.path.exists(cache_dir):
    #         os.makedirs(cache_dir)
    #     mem = Memory(cachedir=cache_dir, verbose=5)

    #     # deleteorient for func
    #     for attr in ['n_sessions', 'session_output_dirs']:
    #         if getattr(self, attr) is None:
    #             warnings.warn("'%s' attribute of is None! Skipping" % attr)
    #             break
    #     else:
    #         self.func = [mem.cache(delete_orientation)(
    #             self.func[sess], self.session_output_dirs[sess])
    #                      for sess in range(self.n_sessions)]

    #     # deleteorient for anat
    #     if self.anat is not None:
    #         self.anat = mem.cache(delete_orientation)(
    #             self.anat, self.anat_output_dir)

    def _sanitize_output_dir(self, output_dir):
        if output_dir is not None:
            output_dir = os.path.abspath(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        return output_dir

    def _sanitize_session_output_dirs(self):
        """func output dirs (one per session)"""
        if self.session_output_dirs is None:
            if self.n_sessions is None:
                return
            self.session_output_dirs = [None] * self.n_sessions

        # session-wise func output directories
        for sess, sess_output_dir in enumerate(self.session_output_dirs):
            if sess_output_dir is None:
                if self.n_sessions > 1:
                    sess_output_dir = os.path.join(
                        self.output_dir, self.session_ids[sess])
                else:
                    sess_output_dir = self.output_dir
            else:
                sess_output_dir = sess_output_dir
            self.session_output_dirs[sess] = self._sanitize_output_dir(
                sess_output_dir)

    def _sanitize_session_scratch_dirs(self):
        """func scratch dirs (one per session)"""
        if self.session_scratch_dirs is None:
            if self.n_sessions is None:
                return
            self.session_scratch_dirs = [None] * self.n_sessions

        # session-wise func scratch directories
        for sess, sess_scratch_dir in enumerate(self.session_scratch_dirs):
            if sess_scratch_dir is None:
                if self.n_sessions > 1:
                    sess_scratch_dir = os.path.join(
                        self.scratch, self.session_ids[sess])
                else:
                    sess_scratch_dir = self.scratch
            self.session_scratch_dirs[sess] = self._sanitize_output_dir(
                sess_scratch_dir)

    def _sanitize_output_dirs(self):
        # output dir
        self.output_dir = self._sanitize_output_dir(self.output_dir)

        # anat output dir
        if self.anat_output_dir is None:
            self.anat_output_dir = self.output_dir
        self.anat_output_dir = self._sanitize_output_dir(self.anat_output_dir)

        # sanitize per-session func output dirs
        self._sanitize_session_output_dirs()

    def _sanitize_scratch_dirs(self):
        # scratch dir
        self.scratch = self._sanitize_output_dir(self.scratch)

        # anat scratch dir
        if self.anat_scratch_dir is None:
            self.anat_scratch_dir = self.scratch
        self.anat_scratch_dir =\
            self._sanitize_output_dir(self.anat_scratch_dir)

        # sanitize per-session func scratch dirs
        self._sanitize_session_scratch_dirs()

    def _niigz2nii(self):
        """
        Convert .nii.gz to .nii (crucial for SPM).

        """
        if self.scratch is None:
            self.scratch = self.output_dir
        cache_dir = os.path.join(self.scratch, 'cache_dir')
        mem = Memory(cache_dir, verbose=100)
        self._sanitize_session_output_dirs()
        self._sanitize_session_scratch_dirs()
        if None not in [self.func, self.n_sessions,
                        self.session_scratch_dirs]:
            self.func = [mem.cache(do_niigz2nii)(
                self.func[sess], output_dir=self.session_scratch_dirs[sess])
                for sess in range(self.n_sessions)]
        if self.anat is not None:
            self.anat = mem.cache(do_niigz2nii)(
                self.anat, output_dir=self.anat_scratch_dir)


    # def _dcm2nii(self):
    #     """
    #     Convert DICOM to nifti.

    #     """
    #     self.isdicom = False
    #     if self.func:
    #         if not isinstance(self.func[0], _basestring):
    #             if not is_niimg(self.func[0]):
    #                 self.isdicom = isdicom(self.func[0][0])
    #         self.func = [do_dcm2nii(sess_func, output_dir=self.output_dir)[0]
    #                      for sess_func in self.func]
    #     if self.anat:
    #         self.anat = do_dcm2nii(self.anat, output_dir=self.output_dir)[0]

    def _check_func_names_and_shapes(self):
        """
        Checks that abspaths of func images are distinct with and across
        sessions, and each session should constitute a 4D film (as a
        string or list of)

        """
        # check that functional image abspaths are distinct across sessions
        for sess1 in range(self.n_sessions):
            if is_niimg(self.func[sess1]):
                continue

            # functional images for this session must all be distinct abspaths
            if not isinstance(self.func[sess1], _basestring):
                if len(self.func[sess1]) != len(set(self.func[sess1])):
                    # Oops! there must be a repetition somewhere
                    for x in self.func[sess1]:
                        count = self.func[sess1].count(x)
                        if count > 1:
                            rep = x
                            break
                    raise RuntimeError(
                        "List of functional images for session number %i"
                        " has the file %s repeated %s times" % (
                            sess1 + 1, rep, count))

            # all functional data for this session should constitute a 4D film
            if isinstance(self.func[sess1], _basestring):
                if not is_4D(self.func[sess1]):
                    warnings.warn(
                        "Functional images for session number %i"
                        " doesn't constitute a 4D film; the shape of the"
                        " session is %s; the images for this session are "
                        "%s" % (sess1 + 1, get_shape(self.func[sess1]),
                                self.func[sess1]))

                    # makes no sense to do tsdiffana on a single-volume
                    self.tsdiffana = False
            else:
                for x in self.func[sess1]:
                    if not is_3D(x):
                        raise RuntimeError(
                            "Image %s of session number %i is not 3D; it "
                            "has shape %s." % (x, sess1 + 1, get_shape(x)))

            # functional images for sess1 shouldn't concide with any functional
            # image of any other session
            for sess2 in range(sess1 + 1, self.n_sessions):
                if is_niimg(self.func[sess2]):
                    continue
                if self.func[sess1] == self.func[sess2]:
                    raise RuntimeError(
                        ('The same image %s specified for session number %i '
                         'and %i' % (self.func[sess1], sess1 + 1,
                                     sess2 + 1)))
                if isinstance(self.func[sess1], _basestring):
                    if self.func[sess1] == self.func[sess2]:
                        raise RuntimeError(
                            'The same image %s specified for session '
                            "number %i and %i" % (
                                self.func[sess1], sess1 + 1, sess2 + 1))
                else:
                    if not isinstance(self.func[sess2], _basestring):
                        if self.func[sess2] in self.func[sess1]:
                            raise RuntimeError(
                                'The same image %s specified for session'
                                ' number %i and %i' % (
                                    self.func[sess1], sess1 + 1, sess2 + 1))
                        else:
                            for x in self.func[sess1]:
                                if is_niimg(x):
                                    continue
                                for y in self.func[sess2]:
                                    if is_niimg(y):
                                        continue
                                    if x == y:
                                        raise RuntimeError(
                                            'The same image %s specified for '
                                            'in both session number %i '
                                            'and %i' % (x, sess1 + 1,
                                                        sess2 + 1))

    def _set_session_ids(self):
        if self.func is None:
            return
        elif isinstance(self.func, _basestring): self.func = [self.func]
        if self.session_ids is None:
            if len(self.func) > 10:
                warnings.warn(
                    "There are more than 10 sessions for subject"
                    " %s. Are you sure something is not wrong ?" % (
                        self.subject_id))
            self.session_ids = ["Session%i" % (sess + 1)
                                for sess in range(len(self.func))]
        else:
            if isinstance(self.session_ids, (_basestring, int)):
                assert len(self.func) == 1
                self.session_ids = [self.session_ids]
            else:
                assert len(self.session_ids) == len(self.func), "%s != %s" % (
                    len(self.session_ids), len(self.func))
        self.n_sessions = len(self.session_ids)

    def sanitize(self, 
                niigz2nii=False):
        """
        This method does basic sanitization of the `SubjectData` instance, like
        extracting .nii.gz -> .nii (crucial for SPM), ensuring that functional
        images actually exist on disk, etc.

        Parameters
        ----------
        niigz2nii: bool, optional (default False)
            convert func and ant .nii.gz images to .nii

        """
        # sanitize output_dir
        self._sanitize_output_dirs()

        # sanitize func
        if isinstance(self.func, _basestring):
            self.func = [self.func]

        # sanitize anat
        if not self.anat is None:
            if not os.path.isfile(self.anat):
                raise OSError("%s is not a file!" % self.anat)

        # sanitize session_ids
        if self.func is None:
            return self
        self._set_session_ids()

        # .nii.gz -> .nii extraction for SPM 
        if niigz2nii:
            self._niigz2nii()

        self._check_func_names_and_shapes()

        # get basenames
        self.basenames = [get_basenames(self.func[sess]) if not is_niimg(
                self.func[sess]) else "%s.nii.gz" % self.session_ids[sess]
                        for sess in range(self.n_sessions)]

        self._set_session_ids()
        return self

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)
