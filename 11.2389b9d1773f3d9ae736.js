(window.webpackJsonp=window.webpackJsonp||[]).push([[11],{"/2EW":function(t,o,i){"use strict";i.r(o),i.d(o,"NotificationModule",function(){return O});var e=i("Xa2L"),n=i("/1cH"),a=i("iadO"),l=i("1jcm"),r=i("d3UM"),c=i("NFeN"),s=i("qFsG"),u=i("kmnG"),b=i("0IaG"),d=i("V5BG"),p=i("Q4Mo"),f=i("jIHw"),m=i("7kUa"),h=i("3Pt+"),g=i("rEr+"),v=i("PCNd"),y=i("ofXK"),w=i("tyNb"),k=i("mrSG"),C=i("12jx"),V=i("4ZtF"),W=i("nnAt"),D=i("fXoL"),S=i("otk6"),M=i("FKr1");function N(t,o){if(1&t&&(D.Wb(0,"mat-error",8),D.Qc(1),D.Vb()),2&t){const t=D.hc();D.Cb(1),D.Rc(t.displayMessage.title)}}let Q=(()=>{class t{constructor(t,o,i){var e,n,a,l,r,c,s,u,b,d,p,f;this.fb=t,this.fun=o,this.config=i,this.formSubmit=new D.o,this.formClose=new D.o,this.isPublished=!1,this.displayMessage={},this.category=null===(e=this.config.lookupData)||void 0===e?void 0:e.category,this.image=Object(V.k)(null===(n=this.config.formData)||void 0===n?void 0:n.image)?null===(a=this.config.formData)||void 0===a?void 0:a.image:Object(V.p)(null===(l=this.config.formData)||void 0===l?void 0:l.image)?[null===(r=this.config.formData)||void 0===r?void 0:r.image]:[];const m=null===(s=null===(c=this.config.formData)||void 0===c?void 0:c.date)||void 0===s?void 0:s.split(":");null==m||m.pop(),this.form=this.fb.group({id:null===(u=this.config.formData)||void 0===u?void 0:u.id,Title:[null===(b=this.config.formData)||void 0===b?void 0:b.Title,h.v.required],category:(null===(d=this.config.formData)||void 0===d?void 0:d.category)||"SYSTEM",content:null===(p=this.config.formData)||void 0===p?void 0:p.content,date:(null==m?void 0:m.join(":"))||"",created_at:null===(f=this.config.formData)||void 0===f?void 0:f.created_at}),this.validationMessages={title:{required:"field is required."}},this.genericValidator=new W.a(this.validationMessages)}ngOnInit(){this.form.valueChanges.subscribe(()=>this.displayMessage=this.genericValidator.processMessages(this.form))}blur(){this.displayMessage=this.genericValidator.processMessages(this.form)}keyDown(t,o){return t.key?this.fun.allowedKey(t.key,o):(t.target.value=this.fun.removeNotAllowedKey(t.target.value,o),!0)}getFile(t){this.form.controls.image.setValue((null==t?void 0:t.upload.length)>0?null==t?void 0:t.upload:[])}onSubmit(){const t=this.form.value;t.date=new Date(t.date||new Date),this.formSubmit.emit(t)}onCancel(){this.formClose.emit()}}return t.\u0275fac=function(o){return new(o||t)(D.Qb(h.e),D.Qb(S.a),D.Qb(b.a))},t.\u0275cmp=D.Kb({type:t,selectors:[["app-notification-form"]],outputs:{formSubmit:"formSubmit",formClose:"formClose"},decls:45,vars:4,consts:[["autocomplete","off",3,"formGroup","submit"],["mat-dialog-title","",1,"p-dialog-header","p-d-flex","p-jc-lg-between",2,"padding","0.2re 0.3rem"],[1,"p-dialog-title","capitalize"],["pButton","","pRipple","","icon","pi pi-times",1,"shadow-none","p-button-rounded","p-button-plain","p-button-text","p-mr-1",3,"click"],["mat-dialog-content","",1,"p-dialog-content"],[1,"p-grid"],[1,"p-col-12"],[1,"full-width"],[1,"error"],["matInput","","tabindex","0","type","text","value","","formControlName","Title",3,"blur","input","keydown"],["class","error",4,"ngIf"],["formControlName","category",3,"blur"],["value","SYSTEM"],["value","APP"],["matInput","","tabindex","0","rows","3","type","text","value","","formControlName","content",3,"blur","input","keydown"],["matInput","","tabindex","0","type","datetime-local","value","","formControlName","date"],["mat-dialog-actions","",1,"p-dialog-footer","button-row"],["pbutton","","pripple","","label","Cancel","type","button","icon","pi pi-times",1,"p-button-text","p-ripple","p-button","p-component","shadow-none","p-mr-2",3,"click"],["aria-hidden","true",1,"p-button-icon","p-button-icon-left","pi","pi-times"],[1,"p-button-label"],[1,"p-ink"],["pbutton","","pripple","","label","Save","icon","pi pi-check","type","submit",1,"p-button-text","p-ripple","p-button","p-component","shadow-none",3,"disabled"],["aria-hidden","true",1,"p-button-icon","p-button-icon-left","pi","pi-check"]],template:function(t,o){1&t&&(D.Wb(0,"form",0),D.ec("submit",function(){return o.onSubmit()}),D.Wb(1,"div",1),D.Wb(2,"div",2),D.Qc(3),D.Vb(),D.Wb(4,"button",3),D.ec("click",function(){return o.onCancel()}),D.Vb(),D.Vb(),D.Wb(5,"div",4),D.Wb(6,"div",5),D.Wb(7,"div",6),D.Wb(8,"mat-form-field",7),D.Wb(9,"mat-label"),D.Qc(10,"Title"),D.Wb(11,"span",8),D.Qc(12,"*"),D.Vb(),D.Vb(),D.Wb(13,"input",9),D.ec("blur",function(){return o.blur()})("input",function(t){return o.keyDown(t,"text")})("keydown",function(t){return o.keyDown(t,"text")}),D.Vb(),D.Oc(14,N,2,1,"mat-error",10),D.Vb(),D.Vb(),D.Wb(15,"div",6),D.Wb(16,"mat-form-field",7),D.Wb(17,"mat-label"),D.Qc(18,"Display Method"),D.Vb(),D.Wb(19,"mat-select",11),D.ec("blur",function(){return o.blur()}),D.Wb(20,"mat-option",12),D.Qc(21,"SYSTEM"),D.Vb(),D.Wb(22,"mat-option",13),D.Qc(23,"APP"),D.Vb(),D.Vb(),D.Vb(),D.Vb(),D.Wb(24,"div",6),D.Wb(25,"mat-form-field",7),D.Wb(26,"mat-label"),D.Qc(27,"Content"),D.Vb(),D.Wb(28,"textarea",14),D.ec("blur",function(){return o.blur()})("input",function(t){return o.keyDown(t,"text")})("keydown",function(t){return o.keyDown(t,"text")}),D.Vb(),D.Vb(),D.Vb(),D.Wb(29,"div",6),D.Wb(30,"mat-form-field",7),D.Wb(31,"mat-label"),D.Qc(32,"Show At"),D.Vb(),D.Rb(33,"input",15),D.Vb(),D.Vb(),D.Vb(),D.Vb(),D.Wb(34,"div",16),D.Wb(35,"button",17),D.ec("click",function(){return o.onCancel()}),D.Rb(36,"span",18),D.Wb(37,"span",19),D.Qc(38,"Cancel"),D.Vb(),D.Rb(39,"span",20),D.Vb(),D.Wb(40,"button",21),D.Rb(41,"span",22),D.Wb(42,"span",19),D.Qc(43,"Save"),D.Vb(),D.Rb(44,"span",20),D.Vb(),D.Vb(),D.Vb()),2&t&&(D.oc("formGroup",o.form),D.Cb(3),D.Rc(o.config.title),D.Cb(11),D.oc("ngIf",o.displayMessage.title),D.Cb(26),D.oc("disabled",!o.form.valid))},directives:[h.w,h.q,h.h,b.g,f.b,p.a,b.e,u.c,u.g,s.b,h.c,h.p,h.f,y.m,r.a,M.h,b.c,u.b],styles:[""]}),t})();var x=i("sSZD"),B=i("wD+u");let I=(()=>{class t{constructor(t,o,i){this.db=t,this.firestore=o,this.fun=i,this.url="/Notification"}get(t=null){const o=`${this.url}`;return t?this.firestore.collection(o).doc(t):this.firestore.collection(o)}add(t,o=null){return delete t.id,this.firestore.collection(this.url).add(t).then(t=>{var i;null==o||o.ref.close(),null===(i=null==o?void 0:o.block)||void 0===i||i.stop()}).catch(t=>{var i;null===(i=null==o?void 0:o.block)||void 0===i||i.stop()})}update(t,o,i=null){return delete o.id,this.firestore.collection(this.url).doc(t).update(o).then(t=>{null==i||i.ref.close(),null==i||i.block.stop()}).catch(t=>{var o;null===(o=null==i?void 0:i.block)||void 0===o||o.stop()})}delete(t){return this.firestore.collection(this.url).doc(t).delete()}}return t.\u0275fac=function(o){return new(o||t)(D.ac(x.a),D.ac(B.a),D.ac(S.a))},t.\u0275prov=D.Mb({token:t,factory:t.\u0275fac,providedIn:"root"}),t})();var A=i("H0VJ"),j=i("WLRH");const T=[{path:"",component:(()=>{class t{constructor(t,o){this.notificationService=t,this.dialogServices=o,this.caption="Notifications",this.columns=[{label:"Title",name:"Title",sortable:!0},{label:"Content",name:"content",sortable:!0},{label:"Category",name:"category",sortable:!0},{label:"Show At",name:"date",type:"date",sortable:!0}],this.actions=[{icon:"pencil",color:"warning",disable:!1},{icon:"trash",color:"danger"}],this.toolBarActions=[{position:"right",action:[]},{position:"left",action:[{label:"Create",icon:"plus",color:"",tooltip:null}]}],this.dialogConfig={width:"450px",formComponent:Q,service:this.notificationService}}ngOnInit(){let t;this.blockUI.start("Loading..."),this.notification$=t=this.notificationService.get().snapshotChanges(),t.subscribe(this.blockUI.stop())}add(){this.dialogConfig.title="New Notification",this.dialogConfig.formData="",this.dialogServices.handleDialog(this.dialogConfig)}update(t){this.dialogConfig.title="Edit Notification",this.dialogConfig.formData=t,this.dialogServices.handleDialog(this.dialogConfig)}onActionClick(t){var o;"pencil"===t.type?this.update(t.data):"trash"===t.type&&(this.blockUI.start("Deleting..."),null===(o=this.notificationService.delete(t.data.id))||void 0===o||o.then(this.blockUI.stop()))}onToolBarActionClick(t){"plus"===t&&this.add()}}return t.\u0275fac=function(o){return new(o||t)(D.Qb(I),D.Qb(A.a))},t.\u0275cmp=D.Kb({type:t,selectors:[["app-notification"]],decls:2,vars:10,consts:[[3,"caption","columns","data","actions","first","rows","sortBy","toolBarActions","buttonClick","toolBarButtonClick"]],template:function(t,o){1&t&&(D.Wb(0,"app-table-template",0),D.ec("buttonClick",function(t){return o.onActionClick(t)})("toolBarButtonClick",function(t){return o.onToolBarActionClick(t)}),D.ic(1,"async"),D.Vb()),2&t&&D.oc("caption",o.caption)("columns",o.columns)("data",D.jc(1,8,o.notification$))("actions",o.actions)("first",(null==o.currentPage?null:o.currentPage.first)||0)("rows",(null==o.currentPage?null:o.currentPage.rows)||10)("sortBy",o.sortBy)("toolBarActions",o.toolBarActions)},directives:[j.a],pipes:[y.b],styles:[""]}),Object(k.__decorate)([Object(C.a)()],t.prototype,"blockUI",void 0),t})()},{path:"**",redirectTo:""}];let P=(()=>{class t{}return t.\u0275mod=D.Ob({type:t}),t.\u0275inj=D.Nb({factory:function(o){return new(o||t)},imports:[[w.e.forChild(T)],w.e]}),t})(),O=(()=>{class t{}return t.\u0275mod=D.Ob({type:t}),t.\u0275inj=D.Nb({factory:function(o){return new(o||t)},imports:[[y.c,P,v.a,g.f,h.j,h.u,m.b,f.c,p.b,d.a,b.f,u.e,s.c,c.a,r.b,l.b,a.a,n.a,e.a]]}),t})()}}]);