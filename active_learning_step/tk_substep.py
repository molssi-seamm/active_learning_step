# -*- coding: utf-8 -*-

"""The graphical part shared by the sub-steps of the Active Learning step."""

import tkinter as tk
import tkinter.ttk as ttk

import seamm
import seamm_widgets as sw

#: Parameters laid out by other means, or not at all
_SKIP = ("results", "extra keywords", "create tables")


class TkSubstep(seamm.TkNode):
    """A sub-step's dialog: its parameters, shown only when they apply.

    Subclasses set :attr:`dialog_title`, list in :attr:`triggers` the choices
    that change which parameters apply, and say which do in :meth:`visible`, so
    a setting that does not apply to the current choices is never shown.

    Attributes
    ----------
    dialog_title : str
        The title of the dialog.
    triggers : tuple of str
        The parameters whose change lays the dialog out again.
    """

    dialog_title = "Active Learning"
    triggers = ()

    def __init__(
        self,
        tk_flowchart=None,
        node=None,
        canvas=None,
        x=None,
        y=None,
        w=200,
        h=50,
    ):
        self.dialog = None

        super().__init__(
            tk_flowchart=tk_flowchart,
            node=node,
            canvas=canvas,
            x=x,
            y=y,
            w=w,
            h=h,
        )

    @property
    def structure_selection(self):
        """Whether the step has the standard structure-selection parameters."""
        return "source systems" in self.node.parameters

    def create_dialog(self):
        """Create the dialog, with a Parameters tab and the standard Results tab."""
        super().create_dialog(
            title=self.dialog_title, widget="notebook", results_tab=True
        )

        P = self.node.parameters

        frame = self["parameters frame"] = ttk.LabelFrame(
            self["frame"],
            borderwidth=4,
            relief="sunken",
            text=f"{self.dialog_title} Parameters",
            labelanchor="n",
            padding=10,
        )

        selection = seamm.standard_parameters.structure_selection_parameters
        if self.structure_selection:
            self.create_structure_selection_widgets(frame)
        for key in P:
            if key[0] == "_" or key in _SKIP or key in selection:
                continue
            self[key] = P[key].widget(frame)
        for key in self.triggers:
            for event in ("<<ComboboxSelected>>", "<Return>", "<FocusOut>"):
                self[key].combobox.bind(event, self.reset_dialog)

        self.reset_dialog()

    def reset_dialog(self, widget=None):
        """Lay out the parameters that apply to the current choices."""
        frame = self["frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()
        self["parameters frame"].grid(row=0, column=0, sticky=tk.EW, pady=10)
        frame.columnconfigure(0, weight=1)

        frame = self["parameters frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        selection = seamm.standard_parameters.structure_selection_parameters
        row = 0
        widgets = []
        for key in self.node.parameters:
            if key[0] == "_" or key in _SKIP:
                continue
            if key in selection:
                if key == "source systems" and self.visible(key):
                    row, choices = self.layout_structure_selection(row=row)
                    widgets.extend(choices)
                continue
            if not self.visible(key):
                continue
            self[key].grid(row=row, column=0, columnspan=2, sticky=tk.EW)
            widgets.append(self[key])
            row += 1

        sw.align_labels(widgets, sticky=tk.E)
        frame.columnconfigure(1, weight=1)
        return 1

    def visible(self, key):
        """Whether the parameter ``key`` applies to the current choices."""
        return True

    def value(self, key):
        """The current value of a parameter's widget, as text."""
        return str(self[key].get()).strip()

    def right_click(self, event):
        """Handle a right-click: add the Edit... item and post the menu."""
        super().right_click(event)
        self.popup_menu.add_command(label="Edit..", command=self.edit)
        self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
