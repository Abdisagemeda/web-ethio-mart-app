(window.webpackJsonp=window.webpackJsonp||[]).push([[12],{AVRW:function(t,e,o){"use strict";o.r(e),o.d(e,"UsersModule",function(){return N});var i=o("Xa2L"),n=o("/1cH"),a=o("iadO"),s=o("1jcm"),l=o("d3UM"),r=o("NFeN"),c=o("qFsG"),b=o("kmnG"),u=o("0IaG"),d=o("V5BG"),p=o("Q4Mo"),m=o("jIHw"),f=o("7kUa"),h=o("3Pt+"),g=o("rEr+"),v=o("PCNd"),C=o("ofXK"),y=o("mrSG"),k=o("4ZtF"),w=o("nnAt"),V=o("fXoL"),W=o("otk6"),M=o("FKr1");const j=function(){return{standalone:!0}};let B=(()=>{class t{constructor(t,e,o){var i,n,a,s,l,r,c,b,u,d;this.fb=t,this.fun=e,this.config=o,this.formSubmit=new V.o,this.formClose=new V.o,this.isPublished=!1,this.displayMessage={},this.category=null===(i=this.config.lookupData)||void 0===i?void 0:i.category,this.image=Object(k.j)(null===(n=this.config.formData)||void 0===n?void 0:n.image)?null===(a=this.config.formData)||void 0===a?void 0:a.image:Object(k.n)(null===(s=this.config.formData)||void 0===s?void 0:s.image)?[null===(l=this.config.formData)||void 0===l?void 0:l.image]:[],this.isPublished=void 0===(null===(r=this.config.formData)||void 0===r?void 0:r.disabled)||!(null===(c=this.config.formData)||void 0===c?void 0:c.disabled),this.form=this.fb.group({id:null===(b=this.config.formData)||void 0===b?void 0:b.id,role:(null===(u=this.config.formData)||void 0===u?void 0:u.role)||"member",disabled:[this.isPublished,h.v.required],created_at:null===(d=this.config.formData)||void 0===d?void 0:d.created_at}),this.validationMessages={name:{required:"field is required."}},this.genericValidator=new w.a(this.validationMessages)}ngOnInit(){this.form.valueChanges.subscribe(()=>this.displayMessage=this.genericValidator.processMessages(this.form))}blur(){this.displayMessage=this.genericValidator.processMessages(this.form)}keyDown(t,e){return t.key?this.fun.allowedKey(t.key,e):(t.target.value=this.fun.removeNotAllowedKey(t.target.value,e),!0)}getFile(t){this.form.controls.image.setValue((null==t?void 0:t.upload.length)>0?null==t?void 0:t.upload:[])}onSubmit(){const t=Object.assign(Object.assign({},this.form.value),{disabled:!this.isPublished});this.formSubmit.emit(t)}onCancel(){this.formClose.emit()}}return t.\u0275fac=function(e){return new(e||t)(V.Qb(h.e),V.Qb(W.a),V.Qb(u.a))},t.\u0275cmp=V.Kb({type:t,selectors:[["app-users-form"]],outputs:{formSubmit:"formSubmit",formClose:"formClose"},decls:33,vars:6,consts:[["autocomplete","off",3,"formGroup","submit"],["mat-dialog-title","",1,"p-dialog-header","p-d-flex","p-jc-lg-between",2,"padding","0.2re 0.3rem"],[1,"p-dialog-title","capitalize"],["pButton","","pRipple","","icon","pi pi-times",1,"shadow-none","p-button-rounded","p-button-plain","p-button-text","p-mr-1",3,"click"],["mat-dialog-content","",1,"p-dialog-content"],[1,"p-grid"],[1,"p-col-12"],[1,"full-width"],["formControlName","role",3,"blur"],["value","admin"],["value","member"],["color","primary","forControlName","disabled",1,"example-section",3,"ngModel","ngModelOptions","ngModelChange"],["mat-dialog-actions","",1,"p-dialog-footer","button-row"],["pbutton","","pripple","","label","Cancel","type","button","icon","pi pi-times",1,"p-button-text","p-ripple","p-button","p-component","shadow-none","p-mr-2",3,"click"],["aria-hidden","true",1,"p-button-icon","p-button-icon-left","pi","pi-times"],[1,"p-button-label"],[1,"p-ink"],["pbutton","","pripple","","label","Save","icon","pi pi-check","type","submit",1,"p-button-text","p-ripple","p-button","p-component","shadow-none",3,"disabled"],["aria-hidden","true",1,"p-button-icon","p-button-icon-left","pi","pi-check"]],template:function(t,e){1&t&&(V.Wb(0,"form",0),V.ec("submit",function(){return e.onSubmit()}),V.Wb(1,"div",1),V.Wb(2,"div",2),V.Qc(3),V.Vb(),V.Wb(4,"button",3),V.ec("click",function(){return e.onCancel()}),V.Vb(),V.Vb(),V.Wb(5,"div",4),V.Wb(6,"div",5),V.Wb(7,"div",6),V.Wb(8,"mat-form-field",7),V.Wb(9,"mat-label"),V.Qc(10,"Role"),V.Vb(),V.Wb(11,"mat-select",8),V.ec("blur",function(){return e.blur()}),V.Wb(12,"mat-option",9),V.Qc(13,"Admin"),V.Vb(),V.Wb(14,"mat-option",10),V.Qc(15,"Member"),V.Vb(),V.Vb(),V.Vb(),V.Vb(),V.Wb(16,"mat-slide-toggle",11),V.ec("ngModelChange",function(t){return e.isPublished=t}),V.Wb(17,"span"),V.Qc(18,"Access"),V.Vb(),V.Rb(19,"br"),V.Wb(20,"mat-hint"),V.Qc(21,"block or allow users"),V.Vb(),V.Vb(),V.Vb(),V.Vb(),V.Wb(22,"div",12),V.Wb(23,"button",13),V.ec("click",function(){return e.onCancel()}),V.Rb(24,"span",14),V.Wb(25,"span",15),V.Qc(26,"Cancel"),V.Vb(),V.Rb(27,"span",16),V.Vb(),V.Wb(28,"button",17),V.Rb(29,"span",18),V.Wb(30,"span",15),V.Qc(31,"Save"),V.Vb(),V.Rb(32,"span",16),V.Vb(),V.Vb(),V.Vb()),2&t&&(V.oc("formGroup",e.form),V.Cb(3),V.Rc(e.config.title),V.Cb(13),V.oc("ngModel",e.isPublished)("ngModelOptions",V.rc(5,j)),V.Cb(12),V.oc("disabled",!e.form.valid))},directives:[h.w,h.q,h.h,u.g,m.b,p.a,u.e,b.c,b.g,l.a,h.p,h.f,M.h,s.a,h.s,b.f,u.c],styles:[".example-section[_ngcontent-%COMP%]{display:flex;align-content:center;align-items:center;height:60px}"]}),t})();var P=o("12jx"),S=o("SRE/"),A=o("H0VJ"),D=o("WLRH");let O=(()=>{class t{constructor(t,e){this.userService=t,this.dialogServices=e,this.caption="Users",this.columns=[{label:"Name",name:"name",sortable:!0},{label:"Email",name:"email",type:"email",sortable:!0},{label:"Phone",name:"Phone_number",type:"phone",sortable:!0},{label:"Role",name:"role",sortable:!0},{label:"Access",name:"disabled",type:"status",sortable:!0}],this.actions=[{icon:"cog",disable:!1}],this.toolBarActions=[],this.dialogConfig={width:"400px",formComponent:B,service:this.userService}}ngOnInit(){let t;this.blockUI.start("Loading..."),this.user$=t=this.userService.get().snapshotChanges(),t.subscribe(this.blockUI.stop())}update(t){this.dialogConfig.title="Manage permission",this.dialogConfig.formData=t,this.dialogConfig.lookupData={},this.dialogServices.handleDialog(this.dialogConfig)}onActionClick(t){"cog"===t.type&&this.update(t.data)}onToolBarActionClick(t){}}return t.\u0275fac=function(e){return new(e||t)(V.Qb(S.a),V.Qb(A.a))},t.\u0275cmp=V.Kb({type:t,selectors:[["app-users"]],decls:2,vars:10,consts:[[3,"caption","columns","data","actions","first","rows","sortBy","toolBarActions","buttonClick","toolBarButtonClick"]],template:function(t,e){1&t&&(V.Wb(0,"app-table-template",0),V.ec("buttonClick",function(t){return e.onActionClick(t)})("toolBarButtonClick",function(t){return e.onToolBarActionClick(t)}),V.ic(1,"async"),V.Vb()),2&t&&V.oc("caption",e.caption)("columns",e.columns)("data",V.jc(1,8,e.user$))("actions",e.actions)("first",(null==e.currentPage?null:e.currentPage.first)||0)("rows",(null==e.currentPage?null:e.currentPage.rows)||10)("sortBy",e.sortBy)("toolBarActions",e.toolBarActions)},directives:[D.a],pipes:[C.b],styles:[""]}),Object(y.__decorate)([Object(P.a)()],t.prototype,"blockUI",void 0),t})();var Q=o("tyNb");const R=[{path:"",component:O},{path:"**",redirectTo:""}];let x=(()=>{class t{}return t.\u0275mod=V.Ob({type:t}),t.\u0275inj=V.Nb({factory:function(e){return new(e||t)},imports:[[Q.e.forChild(R)],Q.e]}),t})(),N=(()=>{class t{}return t.\u0275mod=V.Ob({type:t}),t.\u0275inj=V.Nb({factory:function(e){return new(e||t)},imports:[[C.c,x,v.a,g.f,h.j,h.u,f.b,m.c,p.b,d.a,u.f,b.e,c.c,r.a,l.b,s.b,a.a,n.a,i.a]]}),t})()}}]);