# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode


class BaseTensor(SimpleClass):
    """
    Base tensor class with additional methods for easy manipulation and device handling.
    
    This class provides a unified interface for handling tensors, whether they are on the CPU or GPU, and allows 
    for easy conversion between different formats like NumPy arrays and PyTorch tensors.
    
    Attributes:
        data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
        orig_shape (Tuple[int, int]): Original shape of the image, typically in the format (height, width).
    
    Methods:
        shape: Returns the shape of the underlying data tensor.
        cpu: Returns a copy of the tensor stored in CPU memory.
        numpy: Returns a copy of the tensor as a NumPy array.
        cuda: Moves the tensor to GPU memory, returning a new instance if necessary.
        to: Returns a copy of the tensor with the specified device and dtype.
        __len__: Returns the length of the underlying data tensor.
        __getitem__: Returns a new BaseTensor instance containing the specified indexed elements of the data tensor.
    
    Examples:
        Create a BaseTensor instance and perform operations
        >>> import torch
        >>> from ultralytics.engine.results import BaseTensor
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> print(base_tensor.shape)
        >>> base_tensor_cpu = base_tensor.cpu()
        >>> base_tensor_np = base_tensor.numpy()
        >>> base_tensor_gpu = base_tensor.cuda()
    """

    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with prediction data and the original shape of the image.
        
        Args:
            data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
            orig_shape (Tuple[int, int]): Original shape of the image, typically in the format (height, width).
        
        Examples:
            Initialize a BaseTensor with torch.Tensor data
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """
        Returns the shape of the underlying data tensor for easier manipulation and device handling.
        
        Returns:
            (torch.Size | tuple): Shape of the underlying data tensor, which can be a torch.Size object or a tuple 
                representing the dimensions of the tensor.
        
        Examples:
            Get the shape of a BaseTensor object
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> shape = base_tensor.shape
            >>> print(shape)
            torch.Size([2, 3])
        """
        return self.data.shape

    def cpu(self):
        """
        Return a copy of the tensor stored in CPU memory.
        
        Returns:
            (torch.Tensor): A copy of the tensor stored in CPU memory.
        
        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> print(cpu_tensor.device)
            cpu
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """
        Returns a copy of the tensor as a numpy array for efficient numerical operations.
        
        Returns:
            (np.ndarray): Copy of the tensor data as a numpy array.
        
        Examples:
            Convert a BaseTensor to a numpy array
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, (720, 1280))
            >>> numpy_array = base_tensor.numpy()
            >>> print(numpy_array)
            [[1 2 3]
             [4 5 6]]
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """
        Moves the tensor to GPU memory, returning a new instance if necessary.
        
        Returns:
            (BaseTensor): A new instance of BaseTensor with the data moved to GPU memory.
        
        Examples:
            Move a tensor to GPU memory
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> base_tensor_gpu = base_tensor.cuda()
        """
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """
        Return a copy of the tensor with the specified device and dtype.
        
        Args:
            *args: Positional arguments specifying the target device and/or dtype.
            **kwargs: Keyword arguments specifying the target device and/or dtype.
        
        Returns:
            (BaseTensor): A new instance of BaseTensor with the tensor moved to the specified device and/or dtype.
        
        Examples:
            Move tensor to GPU:
            >>> base_tensor = BaseTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]), (720, 1280))
            >>> base_tensor_cuda = base_tensor.to('cuda')
        
            Change tensor dtype:
            >>> base_tensor_float = base_tensor.to(torch.float32)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        """
        Return the length of the underlying data tensor.
        
        Returns:
            (int): The number of elements along the first dimension of the data tensor.
        
        Examples:
            Get the length of a BaseTensor instance
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a new BaseTensor instance containing the specified indexed elements of the data tensor.
        
        Args:
            idx (int | slice | List[int]): Index or indices specifying which elements to retrieve from the data tensor.
        
        Returns:
            (BaseTensor): A new BaseTensor instance containing the indexed elements of the original data tensor.
        
        Examples:
            Retrieve a single element from the BaseTensor
            >>> base_tensor = BaseTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]), (720, 1280))
            >>> element = base_tensor[0]
            >>> print(element.data)
            tensor([1, 2, 3])
        
            Retrieve a slice from the BaseTensor
            >>> base_tensor = BaseTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]), (720, 1280))
            >>> slice_tensor = base_tensor[0:2]
            >>> print(slice_tensor.data)
            tensor([[1, 2, 3],
                    [4, 5, 6]])
        """
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass):
    """
    A class for storing and manipulating inference results.
    
    Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array.
        orig_shape (tuple): Original image shape in (height, width) format.
        boxes (Boxes | None): Object containing detection bounding boxes.
        masks (Masks | None): Object containing detection masks.
        probs (Probs | None): Object containing class probabilities for classification tasks.
        keypoints (Keypoints | None): Object containing detected keypoints for each object.
        speed (dict): Dictionary of preprocess, inference, and postprocess speeds (ms/image).
        names (dict): Dictionary of class names.
        path (str): Path to the image file.
    
    Methods:
        update(boxes=None, masks=None, probs=None, obb=None): Updates object attributes with new detection results.
        cpu(): Returns a copy of the Results object with all tensors on CPU memory.
        numpy(): Returns a copy of the Results object with all tensors as numpy arrays.
        cuda(): Returns a copy of the Results object with all tensors on GPU memory.
        to(*args, **kwargs): Returns a copy of the Results object with tensors on a specified device and dtype.
        new(): Returns a new Results object with the same image, path, and names.
        plot(...): Plots detection results on an input image, returning an annotated image.
        show(): Show annotated results to screen.
        save(filename): Save annotated results to file.
        verbose(): Returns a log string for each task, detailing detections and classifications.
        save_txt(txt_file, save_conf=False): Saves detection results to a text file.
        save_crop(save_dir, file_name=Path("im.jpg")): Saves cropped detection images.
        tojson(normalize=False): Converts detection results to JSON format.
    
    Examples:
        Create a Results object and process results
        >>> results = model("path/to/image.jpg")
        >>> for result in results:
        >>>     result_cuda = result.cuda()
        >>>     result_cpu = result.cpu()
    """

    def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, speed=None
    ) -> None:
        """
        Initialize the Results class for storing and manipulating inference results.
        
        Args:
            orig_img (numpy.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (dict): A dictionary of class names.
            boxes (torch.Tensor, optional): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.Tensor, optional): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.Tensor, optional): A 1D tensor of probabilities of each class for classification tasks.
            keypoints (torch.Tensor, optional): A 2D tensor of keypoint coordinates for each detection. For default pose 
                model, Keypoint indices for human body pose estimation are:
                0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
                5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
                9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
                13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
            obb (torch.Tensor, optional): A 2D tensor of oriented bounding box coordinates for each detection.
            speed (dict, optional): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).
        
        Example:
            ```python
            results = model("path/to/image.jpg")
            ```
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """
        Return a Results object for a specific index of inference results.
        
        Args:
            idx (int): Index of the desired result.
        
        Returns:
            (Results): A new Results object containing the inference results for the specified index.
        
        Examples:
            Access the first result from a list of results
            >>> results = model('path/to/image.jpg')
            >>> first_result = results[0]
        """
        return self._apply("__getitem__", idx)

    def __len__(self):
        """
        Return the number of detections in the Results object from a non-empty attribute set (boxes, masks, etc.).
        
        Returns:
            (int): The number of detections in the Results object.
        
        Examples:
            Get the number of detections in the results
            >>> results = model('path/to/image.jpg')
            >>> len(results)
            5
        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        """
        Updates detection results attributes including boxes, masks, probs, and obb with new data.
        
        Args:
            boxes (torch.Tensor | None): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.Tensor | None): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.Tensor | None): A 1D tensor of probabilities of each class for classification tasks.
            obb (torch.Tensor | None): A 2D tensor of oriented bounding box coordinates for each detection.
        
        Examples:
            Update the results with new bounding boxes and masks:
            >>> results = model("path/to/image.jpg")[0]
            >>> new_boxes = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
            >>> new_masks = torch.rand(2, 1, 640, 640)
            >>> results.update(boxes=new_boxes, masks=new_masks)
        """
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)

    def _apply(self, fn, *args, **kwargs):
        """
        Applies a specified function to all non-empty attributes and returns a new Results object with modified attributes.
        
        Args:
            fn (str): The name of the function to apply.
            *args: Variable length argument list to pass to the function.
            **kwargs: Arbitrary keyword arguments to pass to the function.
        
        Returns:
            (Results): A new Results object with attributes modified by the applied function.
        
        Examples:
            Apply the 'cuda' function to move results to GPU memory:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result_cuda = result.cuda()
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """
        Returns a copy of the Results object with all its tensors moved to CPU memory.
        
        Returns:
            (Results): A new Results object with all tensors stored in CPU memory.
        
        Examples:
            >>> results = model("path/to/image.jpg")
            >>> cpu_results = results.cpu()
        """
        return self._apply("cpu")

    def numpy(self):
        """
        Returns a copy of the Results object with all tensors as numpy arrays.
        
        Returns:
            (Results): A new Results object where all tensor attributes (boxes, masks, probs, keypoints, obb) are 
                converted to numpy arrays.
        
        Examples:
            Convert inference results to numpy arrays:
            >>> results = model("path/to/image.jpg")
            >>> numpy_results = results.numpy()
        """
        return self._apply("numpy")

    def cuda(self):
        """
        Moves all tensors in the Results object to GPU memory.
        
        Returns:
            (Results): A new Results object with all tensors moved to GPU memory.
        
        Examples:
            >>> results = model('path/to/image.jpg')
            >>> result = results[0]
            >>> result_cuda = result.cuda()
        """
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """
        Moves all tensors in the Results object to the specified device and dtype.
        
        Args:
            *args: Variable length argument list to specify the target device and dtype.
            **kwargs: Arbitrary keyword arguments to specify the target device and dtype.
        
        Returns:
            (Results): A new Results object with all tensors moved to the specified device and dtype.
        
        Examples:
            Move results to GPU:
            >>> results = model("path/to/image.jpg")
            >>> results_gpu = results.to("cuda")
        
            Move results to CPU:
            >>> results_cpu = results.to("cpu")
        """
        return self._apply("to", *args, **kwargs)

    def new(self):
        """
        Returns a new Results object with the same image, path, names, and speed attributes.
        
        Returns:
            (Results): A new Results object with the same image, path, names, and speed attributes.
        
        Examples:
            Create a new Results object from an existing one
            >>> results = model("path/to/image.jpg")
            >>> new_results = results.new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
    ):
        """
        Plots the detection results on an input RGB image and returns the annotated image.
        
        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float | None): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float | None): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray | None): Plot to another image. If None, plot to the original image.
            im_gpu (torch.Tensor | None): Normalized image in GPU with shape (1, 3, 640, 640), for faster mask plotting.
            kpt_radius (int): Radius of the drawn keypoints.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            masks (bool): Whether to plot the masks.
            probs (bool): Whether to plot classification probability.
            show (bool): Whether to display the annotated image directly.
            save (bool): Whether to save the annotated image to `filename`.
            filename (str | None): Filename to save image to if save is True.
        
        Returns:
            (numpy.ndarray): A numpy array of the annotated image.
        
        Examples:
            >>> from PIL import Image
            >>> from ultralytics import YOLO
            >>> model = YOLO('yolov8n.pt')
            >>> results = model('bus.jpg')  # results list
            >>> for r in results:
            >>>     im_array = r.plot()  # plot a BGR numpy array of predictions
            >>>     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            >>>     im.show()  # show image
            >>>     im.save('results.jpg')  # save image
        """
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        # Plot Pose results
        if self.keypoints is not None:
            for k in reversed(self.keypoints.data):
                annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename)

        return annotator.result()

    def show(self, *args, **kwargs):
        """
        Show the image with annotated inference results.
        
        Args:
            *args: Variable length argument list to pass to the underlying plot function.
            **kwargs: Arbitrary keyword arguments to pass to the underlying plot function.
        
        Examples:
            >>> model = YOLO('yolov8n.pt')
            >>> results = model('path/to/image.jpg')
            >>> for result in results:
            >>>     result.show()
        """
        self.plot(show=True, *args, **kwargs)

    def save(self, filename=None, *args, **kwargs):
        """
        Save annotated inference results image to file.
        
        Args:
            filename (str | None): The filename to save the annotated image. If None, a default filename is generated 
                using the original image path.
        
        Examples:
            Save the annotated results to a specified file:
            >>> results = model('path/to/image.jpg')
            >>> for result in results:
            >>>     result.save('annotated_image.jpg')
        """
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self):
        """
        Returns a log string for each task in the results, detailing detection and classification outcomes.
        
        Returns:
            (str): A log string summarizing the detections and classifications in the results.
        
        Examples:
            >>> results = model('path/to/image.jpg')
            >>> for result in results:
            >>>     log = result.verbose()
            >>>     print(log)
        """
        log_string = ""
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return log_string if probs is not None else f"{log_string}(no detections), "
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """
        Save detection results to a text file.
        
        Args:
            txt_file (str): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.
        
        Returns:
            (str): Path to the saved text file.
        
        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO('yolov8n.pt')
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result.save_txt("output.txt")
        
        Notes:
            - The file will contain one line per detection or classification with the following structure:
                - For detections: `class confidence x_center y_center width height`
                - For classifications: `confidence class_name`
                - For masks and keypoints, the specific formats will vary accordingly.
            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.
            - Existing contents of the file will not be overwritten; new results will be appended.
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """
        Saves cropped detection images to the specified directory.
        
        Args:
            save_dir (str | Path): Directory path where the cropped images should be saved.
            file_name (str | Path): Filename for the saved cropped image.
        
        Notes:
            This function does not support Classify or Oriented Bounding Box (OBB) tasks. It will warn and exit if called for 
            such tasks.
        
        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolov8n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result.save_crop(save_dir="path/to/save/crops", file_name="crop")
        """
        if self.probs is not None:
            LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / f"{Path(file_name)}.jpg",
                BGR=True,
            )

    def summary(self, normalize=False, decimals=5):
        """
        Convert inference results to a summarized dictionary with optional normalization for box coordinates.
        
        Args:
            normalize (bool): Whether to normalize the box coordinates to the range [0, 1].
            decimals (int): Number of decimal places to round the results.
        
        Returns:
            (List[Dict]): A list of dictionaries summarizing the inference results. Each dictionary contains the class name, 
                class ID, confidence score, and bounding box coordinates. If applicable, it also includes tracking ID, 
                segmentation masks, and keypoints.
        
        Examples:
            Summarize inference results with normalized coordinates:
            >>> results = model('path/to/image.jpg')
            >>> summary = results[0].summary(normalize=True)
            >>> print(summary)
            [{'name': 'person', 'class': 0, 'confidence': 0.999, 'box': {'x1': 0.1, 'y1': 0.2, 'x2': 0.3, 'y2': 0.4}}]
        """
        # Create list of detection dictionaries
        results = []
        if self.probs is not None:
            class_id = self.probs.top1
            results.append(
                {
                    "name": self.names[class_id],
                    "class": class_id,
                    "confidence": round(self.probs.top1conf.item(), decimals),
                }
            )
            return results

        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            class_id, conf = int(row.cls), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals named argument required
                    "y": (y / h).numpy().round(decimals).tolist(),
                    "visible": visible.numpy().round(decimals).tolist(),
                }
            results.append(result)

        return results

    def tojson(self, normalize=False, decimals=5):
        """
        Converts detection results to JSON format.
        
        Args:
            normalize (bool): Whether to normalize bounding box coordinates to the range [0, 1].
            decimals (int): Number of decimal places to round the coordinates and confidence scores.
        
        Returns:
            (str): JSON string representing the detection results.
        
        Examples:
            >>> results = model('path/to/image.jpg')
            >>> result = results[0]
            >>> json_str = result.tojson(normalize=True, decimals=4)
            >>> print(json_str)
        """
        import json

        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)


class Boxes(BaseTensor):
    """
    Manages detection boxes, providing easy access and manipulation of box coordinates, confidence scores, class
    identifiers, and optional tracking IDs. Supports multiple formats for box coordinates, including both absolute and
    normalized forms.
    
    Attributes:
        data (torch.Tensor): The raw tensor containing detection boxes and their associated data.
        orig_shape (tuple): The original image size as a tuple (height, width), used for normalization.
        is_track (bool): Indicates whether tracking IDs are included in the box data.
    
    Properties:
        xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | numpy.ndarray, optional): Tracking IDs for each box, if available.
        xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format, calculated on demand.
        xyxyn (torch.Tensor | numpy.ndarray): Normalized [x1, y1, x2, y2] boxes, relative to `orig_shape`.
        xywhn (torch.Tensor | numpy.ndarray): Normalized [x, y, width, height] boxes, relative to `orig_shape`.
    
    Methods:
        cpu(): Moves the boxes to CPU memory.
        numpy(): Converts the boxes to a numpy array format.
        cuda(): Moves the boxes to CUDA (GPU) memory.
        to(device, dtype=None): Moves the boxes to the specified device.
    
    Examples:
        >>> import torch
        >>> from ultralytics.engine.results import Boxes
        >>> boxes_data = torch.tensor([[10, 20, 30, 40, 0.9, 1], [50, 60, 70, 80, 0.8, 2]])
        >>> orig_shape = (720, 1280)
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize the Boxes class with detection box data and the original image shape.
        
        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape (num_boxes, 6) or 
                (num_boxes, 7). Columns should contain [x1, y1, x2, y2, confidence, class, (optional) track_id]. The track ID 
                column is included if present.
            orig_shape (tuple): The original image shape as (height, width). Used for normalization.
        
        Returns:
            (None)
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """
        Returns bounding boxes in [x1, y1, x2, y2] format.
        
        Returns:
            (torch.Tensor | numpy.ndarray): Bounding boxes in [x1, y1, x2, y2] format.
        
        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 1]]), (720, 1280))
            >>> boxes.xyxy
            tensor([[10., 20., 30., 40.]])
        """
        return self.data[:, :4]

    @property
    def conf(self):
        """
        Returns the confidence scores for each detection box.
        
        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the confidence scores for each detection box.
        
        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import Boxes
            >>> boxes_data = torch.tensor([[10, 20, 30, 40, 0.9, 1], [15, 25, 35, 45, 0.8, 2]])
            >>> orig_shape = (720, 1280)
            >>> boxes = Boxes(boxes_data, orig_shape)
            >>> confidence_scores = boxes.conf
            >>> print(confidence_scores)
            tensor([0.9000, 0.8000])
        """
        return self.data[:, -2]

    @property
    def cls(self):
        """
        Returns the class IDs for each detection box.
        
        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the class IDs for each detection box.
        
        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 1], [15, 25, 35, 45, 0.8, 2]]), (720, 1280))
            >>> class_ids = boxes.cls
            >>> print(class_ids)
            tensor([1, 2])
        """
        return self.data[:, -1]

    @property
    def id(self):
        """
        Returns the tracking IDs for each detection box if available.
        
        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the tracking IDs for each detection box. 
            If tracking IDs are not available, returns None.
        
        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 1, 123], [15, 25, 35, 45, 0.8, 2, 124]]), (720, 1280))
            >>> tracking_ids = boxes.id
            >>> print(tracking_ids)
            tensor([123, 124])
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """
        Converts bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.
        
        Returns:
            (torch.Tensor | numpy.ndarray): Bounding boxes in [x, y, width, height] format, with shape (N, 4).
        
        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 1]]), (720, 1280))
            >>> xywh_boxes = boxes.xywh
            >>> print(xywh_boxes)
            tensor([[20., 30., 20., 20.]])
        """
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """
        Normalize box coordinates to [x1, y1, x2, y2] relative to the original image size.
        
        Returns:
            (torch.Tensor | numpy.ndarray): Normalized bounding boxes in [x1, y1, x2, y2] format, relative to the original 
            image size.
        
        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import Boxes
            >>> boxes_data = torch.tensor([[50, 30, 200, 150, 0.9, 1]])
            >>> orig_shape = (300, 400)
            >>> boxes = Boxes(boxes_data, orig_shape)
            >>> normalized_boxes = boxes.xyxyn
            >>> print(normalized_boxes)
            tensor([[0.1250, 0.1000, 0.5000, 0.5000]])
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """
        Returns bounding boxes in normalized [x, y, width, height] format.
        
        Returns:
            (torch.Tensor | numpy.ndarray): Bounding boxes in normalized [x, y, width, height] format, relative to the 
            original image size.
        
        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import Boxes
            >>> boxes_data = torch.tensor([[50, 30, 200, 150, 0.9, 1]])
            >>> orig_shape = (720, 1280)
            >>> boxes = Boxes(boxes_data, orig_shape)
            >>> normalized_boxes = boxes.xywhn
            >>> print(normalized_boxes)
        """
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.
    
    Attributes:
        xy (List): A list of segments in pixel coordinates.
        xyn (List): A list of normalized segments.
    
    Methods:
        cpu(): Returns the masks tensor on CPU memory.
        numpy(): Returns the masks tensor as a numpy array.
        cuda(): Returns the masks tensor on GPU memory.
        to(device, dtype): Returns the masks tensor with the specified device and dtype.
    
    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO('yolov8n.pt')
        >>> results = model('path/to/image.jpg')
        >>> for result in results:
        >>>     masks = result.masks
        >>>     masks_cpu = masks.cpu()
        >>>     masks_cuda = masks.cuda()
    """

    def __init__(self, masks, orig_shape) -> None:
        """
        Initializes the Masks class with a masks tensor and the original image shape.
        
        Args:
            masks (torch.Tensor | np.ndarray): A 3D tensor or numpy array containing detection masks, where each mask is a 
                binary image with shape (H, W) or (N, H, W) for N masks.
            orig_shape (tuple): The original shape of the image, typically in the format (height, width).
        
        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import Masks
            >>> masks = torch.rand(5, 640, 640)  # 5 masks of size 640x640
            >>> orig_shape = (720, 1280)
            >>> masks_obj = Masks(masks, orig_shape)
        """
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """
        Return normalized xy-coordinates of the segmentation masks.
        
        Returns:
            (List[numpy.ndarray]): A list of arrays, each containing the normalized xy-coordinates of a segmentation mask.
        
        Examples:
            >>> masks = Masks(torch.tensor([[[0, 1], [1, 0]]]), (720, 1280))
            >>> normalized_coords = masks.xyn
            >>> print(normalized_coords)
            [array([[0.00078125, 0.00138889],
                    [0.0015625 , 0.00069444]])]
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """
        Returns the [x, y] coordinates of the segmentation masks.
        
        Returns:
            (List[numpy.ndarray]): A list of numpy arrays, each containing the [x, y] coordinates of the segmentation masks.
        
        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO('yolov8n.pt')
            >>> results = model('path/to/image.jpg')
            >>> for result in results:
            >>>     mask_coordinates = result.masks.xy
            >>>     print(mask_coordinates)
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]


class Keypoints(BaseTensor):
    """
    A class for storing and manipulating detection keypoints.
    
    Attributes:
        xy (torch.Tensor): A collection of keypoints containing x, y coordinates for each detection.
        xyn (torch.Tensor): A normalized version of xy with coordinates in the range [0, 1].
        conf (torch.Tensor): Confidence values associated with keypoints if available, otherwise None.
    
    Methods:
        cpu(): Returns a copy of the keypoints tensor on CPU memory.
        numpy(): Returns a copy of the keypoints tensor as a numpy array.
        cuda(): Returns a copy of the keypoints tensor on GPU memory.
        to(device, dtype): Returns a copy of the keypoints tensor with the specified device and dtype.
    
    Examples:
        >>> keypoints = Keypoints(torch.tensor([[[100, 200, 0.9], [150, 250, 0.8]]]), (640, 480))
        >>> keypoints.cpu()
        >>> keypoints.numpy()
        >>> keypoints.cuda()
        >>> keypoints.to('cuda:0')
    """

    @smart_inference_mode()  # avoid keypoints < conf in-place error
    def __init__(self, keypoints, orig_shape) -> None:
        """
        Initializes the Keypoints object with detection keypoints and original image dimensions.
        
        Args:
            keypoints (torch.Tensor | np.ndarray): A tensor or numpy array containing keypoints data with shape 
                (num_detections, num_keypoints, 2) or (num_detections, num_keypoints, 3). The last dimension should 
                contain [x, y] coordinates or [x, y, confidence] values.
            orig_shape (tuple): The original image shape as (height, width). Used for normalization.
        
        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import Keypoints
            >>> keypoints_data = torch.tensor([[[100, 200, 0.9], [150, 250, 0.8]], [[300, 400, 0.7], [350, 450, 0.6]]])
            >>> orig_shape = (720, 1280)
            >>> keypoints = Keypoints(keypoints_data, orig_shape)
        """
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        if keypoints.shape[2] == 3:  # x, y, conf
            mask = keypoints[..., 2] < 0.5  # points with conf < 0.5 (not visible)
            keypoints[..., :2][mask] = 0
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """
        Returns x, y coordinates of keypoints.
        
        Returns:
            (torch.Tensor): A tensor containing the x, y coordinates of keypoints with shape (N, K, 2), where N is the 
                number of detections and K is the number of keypoints per detection.
        
        Examples:
            >>> keypoints = Keypoints(torch.tensor([[[100, 200, 0.9], [150, 250, 0.8]]]), (720, 1280))
            >>> xy = keypoints.xy
            >>> print(xy)
            tensor([[[100, 200],
                     [150, 250]]])
        """
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """
        Returns normalized coordinates (x, y) of keypoints relative to the original image size.
        
        Returns:
            (torch.Tensor): A tensor containing normalized keypoint coordinates with shape (N, K, 2), where N is the number 
                of detections and K is the number of keypoints per detection.
        
        Examples:
            >>> keypoints = torch.tensor([[[100, 200], [150, 250]], [[300, 400], [350, 450]]])
            >>> orig_shape = (800, 600)
            >>> kp = Keypoints(keypoints, orig_shape)
            >>> normalized_kp = kp.xyn
            >>> print(normalized_kp)
            tensor([[[0.1667, 0.2500],
                     [0.2500, 0.3125]],
                    [[0.5000, 0.5000],
                     [0.5833, 0.5625]]])
        """
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        """
        Returns the confidence values associated with keypoints.
        
        Returns:
            (torch.Tensor): A tensor containing confidence values for each keypoint.
        
        Examples:
            >>> keypoints = Keypoints(torch.tensor([[[100, 200, 0.9], [150, 250, 0.8]]]), (640, 480))
            >>> conf = keypoints.conf
            >>> print(conf)
            tensor([[0.9000, 0.8000]])
        """
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):
    """
    A class for storing and manipulating classification predictions.
    
    Attributes:
        top1 (int): Index of the top 1 class.
        top5 (List[int]): Indices of the top 5 classes.
        top1conf (torch.Tensor | np.ndarray): Confidence of the top 1 class.
        top5conf (torch.Tensor | np.ndarray): Confidences of the top 5 classes.
    
    Methods:
        cpu(): Returns a copy of the probs tensor on CPU memory.
        numpy(): Returns a copy of the probs tensor as a numpy array.
        cuda(): Returns a copy of the probs tensor on GPU memory.
        to(device, dtype): Returns a copy of the probs tensor with the specified device and dtype.
    
    Examples:
        >>> import torch
        >>> from ultralytics.engine.results import Probs
        >>> probs = torch.tensor([0.1, 0.3, 0.2, 0.4])
        >>> prob_obj = Probs(probs)
        >>> print(prob_obj.top1)  # Output: 3
        >>> print(prob_obj.top5)  # Output: [3, 1, 2, 0]
        >>> print(prob_obj.top1conf)  # Output: tensor(0.4000)
        >>> print(prob_obj.top5conf)  # Output: tensor([0.4000, 0.3000, 0.2000, 0.1000])
    """

    def __init__(self, probs, orig_shape=None) -> None:
        """
        Initialize Probs with classification probabilities and optional original image shape.
        
        Args:
            probs (torch.Tensor | np.ndarray): A tensor or numpy array containing classification probabilities for each class.
            orig_shape (tuple | None): The original shape of the image, typically in the format (height, width). This is 
                optional and may be None if not applicable.
        
        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import Probs
            >>> probs = torch.tensor([0.1, 0.3, 0.6])
            >>> classification_probs = Probs(probs)
        """
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """
        Return the index of the class with the highest probability.
        
        Returns:
            (int): Index of the class with the highest probability.
        
        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))
            >>> top_class = probs.top1
            >>> print(top_class)
            2
        """
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """
        Retrieve the indices of the top 5 class probabilities.
        
        Returns:
            (List[int]): A list containing the indices of the top 5 class probabilities.
        
        Examples:
            >>> probs = torch.tensor([0.1, 0.3, 0.2, 0.4, 0.05])
            >>> probs_obj = Probs(probs)
            >>> top5_indices = probs_obj.top5
            >>> print(top5_indices)
            [3, 1, 2, 0, 4]
        """
        return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """
        Retrieves the confidence score of the highest probability class.
        
        Returns:
            (torch.Tensor): Confidence score of the top 1 class.
        
        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))
            >>> top1_confidence = probs.top1conf
            >>> print(top1_confidence)
            tensor(0.6000)
        """
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """
        Returns confidence scores for the top 5 classification predictions.
        
        Returns:
            (torch.Tensor): A tensor containing the confidence scores of the top 5 classes.
        
        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.3, 0.2, 0.4, 0.05]))
            >>> top5_confidences = probs.top5conf
            >>> print(top5_confidences)
            tensor([0.4000, 0.3000, 0.2000, 0.1000, 0.0500])
        """
        return self.data[self.top5]


class OBB(BaseTensor):
    """
    A class for storing and manipulating Oriented Bounding Boxes (OBB).
    
    This class provides methods for handling oriented bounding boxes, including conversion to different formats and 
    device handling.
    
    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor containing detection boxes and their associated data.
        orig_shape (tuple): The original image size as a tuple (height, width), used for normalization.
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xywhr (torch.Tensor | numpy.ndarray): The boxes in [x_center, y_center, width, height, rotation] format.
        conf (torch.Tensor | numpy.ndarray): The confidence values of the boxes.
        cls (torch.Tensor | numpy.ndarray): The class values of the boxes.
        id (torch.Tensor | numpy.ndarray, optional): The track IDs of the boxes, if available.
        xyxyxyxy (torch.Tensor | numpy.ndarray): The rotated boxes in xyxyxyxy format.
        xyxyxyxyn (torch.Tensor | numpy.ndarray): The rotated boxes in xyxyxyxy format normalized by original image size.
        xyxy (torch.Tensor | numpy.ndarray): The horizontal boxes in xyxy format.
    
    Methods:
        cpu(): Moves the object to CPU memory.
        numpy(): Converts the object to a numpy array.
        cuda(): Moves the object to CUDA memory.
        to(*args, **kwargs): Moves the object to the specified device.
    
    Examples:
        >>> import torch
        >>> from ultralytics import YOLO
        >>> model = YOLO('yolov8n.pt')
        >>> results = model('path/to/image.jpg')
        >>> for result in results:
        >>>     obb = result.obb
        >>>     if obb is not None:
        >>>         xyxy_boxes = obb.xyxy
        >>>         # Do something with xyxy_boxes
    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        Initializes an OBB (Oriented Bounding Box) instance with detection data and the original image shape.
        
        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array containing the detection boxes, with shape 
                (num_boxes, 7) or (num_boxes, 8). The columns should contain [x_center, y_center, width, height, rotation, 
                confidence, class, (optional) track_id]. The track ID column is included if present.
            orig_shape (tuple): The original image shape as (height, width). Used for normalization.
        
        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import OBB
            >>> boxes = torch.tensor([[100, 150, 50, 75, 0.5, 0.9, 1]])
            >>> orig_shape = (720, 1280)
            >>> obb = OBB(boxes, orig_shape)
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"  # xywh, rotation, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self):
        """
        Returns oriented bounding boxes in [x_center, y_center, width, height, rotation] format.
        
        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array containing the detection boxes, with shape 
                (num_boxes, 7) or (num_boxes, 8). The last two columns contain confidence and class values. If present, 
                the third last column contains track IDs, and the fifth column from the left contains rotation.
            orig_shape (tuple): Original image size, in the format (height, width).
        
        Returns:
            (torch.Tensor | np.ndarray): Oriented bounding boxes in [x_center, y_center, width, height, rotation] format.
        
        Examples:
            >>> import torch
            >>> from ultralytics import OBB
            >>> boxes = torch.tensor([[100, 150, 50, 75, 0.5, 0.9, 1], [200, 250, 60, 80, -0.3, 0.8, 2]])
            >>> orig_shape = (720, 1280)
            >>> obb = OBB(boxes, orig_shape)
            >>> xywhr = obb.xywhr
            >>> print(xywhr)
        """
        return self.data[:, :5]

    @property
    def conf(self):
        """
        Gets the confidence values of Oriented Bounding Boxes (OBBs).
        
        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the confidence values for each OBB.
        
        Examples:
            >>> obb = OBB(torch.tensor([[10, 20, 30, 40, 0.5, 0.9, 1]]), (720, 1280))
            >>> conf = obb.conf
            >>> print(conf)
            tensor([0.9])
        """
        return self.data[:, -2]

    @property
    def cls(self):
        """
        Returns the class values of the oriented bounding boxes.
        
        Args:
            boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes, with shape 
                (num_boxes, 7) or (num_boxes, 8). The last two columns contain confidence and class values. If present, 
                the third last column contains track IDs, and the fifth column from the left contains rotation.
            orig_shape (tuple): Original image size, in the format (height, width).
        
        Returns:
            (torch.Tensor | numpy.ndarray): The class values of the oriented bounding boxes.
        
        Examples:
            >>> import torch
            >>> from ultralytics import OBB
            >>> boxes = torch.tensor([[100, 100, 50, 50, 45, 0.9, 1]])
            >>> orig_shape = (720, 1280)
            >>> obb = OBB(boxes, orig_shape)
            >>> class_values = obb.cls
            >>> print(class_values)
            tensor([1])
        """
        return self.data[:, -1]

    @property
    def id(self):
        """
        Returns the tracking IDs of the oriented bounding boxes (OBBs).
        
        Returns:
            (torch.Tensor | numpy.ndarray | None): Tracking IDs for each OBB if available, otherwise None.
        
        Examples:
            >>> obb_data = torch.tensor([[100, 100, 50, 50, 0.5, 0.9, 1, 123], [200, 200, 60, 60, -0.3, 0.8, 2, 456]])
            >>> orig_shape = (720, 1280)
            >>> obb = OBB(obb_data, orig_shape)
            >>> tracking_ids = obb.id
            >>> print(tracking_ids)
            tensor([123, 456])
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self):
        """
        Convert oriented bounding boxes (OBB) to 8-point (xyxyxyxy) coordinate format.
        
        Returns:
            (torch.Tensor | numpy.ndarray): The oriented bounding boxes in 8-point (xyxyxyxy) format, with shape (N, 8).
        
        Examples:
            >>> obb = OBB(torch.tensor([[50, 50, 40, 20, 0.785, 0.9, 1]]), (100, 100))
            >>> obb.xyxyxyxy
            tensor([[30.0, 40.0, 50.0, 30.0, 70.0, 60.0, 50.0, 70.0]])
        """
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self):
        """
        Converts oriented bounding boxes (OBBs) to normalized 8-point (xyxyxyxy) coordinate format.
        
        Returns:
            (torch.Tensor | numpy.ndarray): Normalized oriented bounding boxes in xyxyxyxy format, with shape (N, 4, 2).
        
        Examples:
            >>> obb = OBB(torch.tensor([[50, 50, 40, 20, 0.5, 0.9, 1]]), (100, 100))
            >>> normalized_boxes = obb.xyxyxyxyn
            >>> print(normalized_boxes)
        """
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self):
        """
        Converts oriented bounding boxes (OBB) to axis-aligned bounding boxes (AABB) in xyxy format.
        
        Returns:
            (torch.Tensor | np.ndarray): Axis-aligned bounding boxes in [x1, y1, x2, y2] format with shape (N, 4).
        
        Examples:
            >>> import torch
            >>> from ultralytics import YOLO
            >>> model = YOLO('yolov8n.pt')
            >>> results = model('path/to/image.jpg')
            >>> for result in results:
            >>>     obb = result.obb
            >>>     if obb is not None:
            >>>         xyxy_boxes = obb.xyxy
            >>>         # Perform operations with xyxy_boxes
        """
        x = self.xyxyxyxy[..., 0]
        y = self.xyxyxyxy[..., 1]
        return (
            torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
            if isinstance(x, torch.Tensor)
            else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
        )
